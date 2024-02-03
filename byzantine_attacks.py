"""
References of IPM, ALIE, LF: https://github.com/epfml/byzantine-robust-optimizer/tree/main/codes/attacks
"""

import torch
import numpy as np
import scipy
import torch.nn as nn
from utils import get_smoothed_and_clipped_gradient, zeros_list_like, get_list_diff_norm, get_clipped_avg


def get_gradient_list(global_model, train_loaders, num_bad_clients, args, device):
    gradient_list = []
    for i in range(num_bad_clients):
        data, target = next(iter(train_loaders[i]))
        data, target = data.to(device), target.to(device)
        gradient_list.append(get_smoothed_and_clipped_gradient(global_model, data, target, args, device))
    return gradient_list 



def get_mean_and_std(gradient_list):

    gradient_mean, gradient_std = [], []
    for idx in range(len(gradient_list[0])):
        layer_gradient_stack = torch.stack([gradient_list[i][idx] for i in range(len(gradient_list))], dim=0)
        gradient_mean.append(torch.mean(layer_gradient_stack, dim=0))
        gradient_std.append(torch.std(layer_gradient_stack, dim=0))
    
    return gradient_mean, gradient_std


def get_list_inner_product(lst1, lst2):
    sum = 0
    for param1, param2 in zip(lst1, lst2):
        sum += torch.dot(torch.flatten(param1), torch.flatten(param2))
    return sum.item()


def byzantine_IPM_or_ALIE(global_model, client_momentums, train_loaders, args, device):
    """
    Inner Product Manipulation (IPM) Attack or A Little Is Enough (ALIE) Attack 
    """
    num_clients = args.num_clients
    num_bad_clients = args.num_bad_clients
    client_selected_rate = args.client_selected_rate


    

    # parameters for ALIE attack
    n = num_clients * client_selected_rate
    m = num_bad_clients * client_selected_rate
    s = np.floor(n / 2 + 1) - m
    cdf_value = min((n - m - s) / (n - m), 0.95)
    z_max = scipy.stats.norm.ppf(cdf_value)
    #print("cdf_value={}, z_max={}".format(cdf_value, z_max))


    gradient_list = get_gradient_list(global_model, train_loaders, num_bad_clients, args, device)
    gradient_mean, gradient_std = get_mean_and_std(gradient_list)


    # bad clients update their momentums
    for idx in range(len(client_momentums[0])):
        if args.attack_type ==  "ipm":
            epsilon = 1  # parameters for IPM attack
            bad_gradient = -epsilon * gradient_mean[idx]
        elif args.attack_type ==  "alie":
            scale = 4  # this huristic scale make larger impact on non-defence aggregation
            bad_gradient = gradient_mean[idx] - scale * gradient_std[idx] * z_max
        else:
            raise NotImplementedError


        for i in range(num_bad_clients):
            client_momentums[i][idx].copy_(bad_gradient)

    return None


def byzantine_LF(global_model, client_momentums, attacker, learn_rate, args, device):
    criterion = nn.CrossEntropyLoss()
    model = attacker.model
    optimizer = attacker.optimizer

    # sync the global model to the attacker's model
    model.load_state_dict(global_model.state_dict())

    # train the attacker's model with 10 steps
    model.train()
    num_iter = 10
    for _ in range(num_iter):
        data, target = next(iter(attacker.train_loader))
        target = 9 - target  # flip the label
        data, target = data.to(device), target.to(device)
    
        model.zero_grad()  
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        # replace the bad clients' momentum by the difference to the global model
        bad_momentum = zeros_list_like(global_model, device)
        for idx, (param_global, param) in enumerate(zip(global_model.parameters(), model.parameters())):

            # Note: DO NOT divided by num_bad_clients (regard it as some gradient)
            bad_momentum[idx].copy_(  (param_global.data - param.data)/learn_rate )  



        for i in range(args.num_bad_clients):
            for idx, param in enumerate(client_momentums[i]):
                param.copy_( bad_momentum[idx] )

    return None



def byzantine_MTB_minmax(client_momentums, attacker, args):
    """"
    Manipulating the Byzantine (MTB) attack [NDSS' 2021]
    paper: https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-3_24498_paper.pdf
    As suggested by the paper, min-max usually use a larger gamma, thus is more effective on the attack
    """
    def get_bad_update(update_mean, deviation, gamma):
        bad_update = []
        for param_update_mean, param_deviation in zip(update_mean, deviation):
            bad_update.append(param_update_mean - gamma*param_deviation)
        return bad_update
    
    
    all_updates = attacker.benign_momentums
    update_mean, update_std = get_mean_and_std(all_updates)

    # Step 1. compute deviation 
    # As suggected by the paper, sign-vector is more effective for MNIST, 
    # while std-vector is more effective for CIFAR-10
    if args.name_dataset == 'mnist':
        deviation = [torch.sign(param) for param in update_mean]
    elif args.name_dataset == 'cifar':
        deviation = update_std


    # Step 2. compute max distance 
    num_updates = len(all_updates)
    distances = []
    for i in range(num_updates):
        for j in range(num_updates):
            if i != j:
                distances.append(get_list_diff_norm(all_updates[i], all_updates[j]))
    max_distance = max(distances)

    
    # Step 3. compute gamma
    gamma = 10
    gamma_fail = gamma
    gamma_succ = 0
    threshold_diff = 1e-5
    while np.abs(gamma_succ - gamma) > threshold_diff:
        bad_update = get_bad_update(update_mean, deviation, gamma)
        distances = [get_list_diff_norm(bad_update, update) for update in all_updates]
        
        if max(distances) <= max_distance:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    gamma = gamma_succ
    #print(f"gamma={gamma}")

    # Step 4. bad clients update their momentums
    bad_momentum = get_bad_update(update_mean, deviation, gamma)
    for idx, param in enumerate(bad_momentum):
        for i in range(args.num_bad_clients):
            client_momentums[i][idx].copy_(param)