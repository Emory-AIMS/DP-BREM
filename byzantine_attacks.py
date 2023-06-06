"""
References: https://github.com/epfml/byzantine-robust-optimizer/tree/main/codes/attacks
"""

import torch
import numpy as np
import scipy
import torch.nn as nn
from utils import get_smoothed_and_clipped_gradient, zeros_list_like



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


    gradient_list = []
    for i in range(num_bad_clients):
        data, target = next(iter(train_loaders[i]))
        data, target = data.to(device), target.to(device)
        gradient_list.append(get_smoothed_and_clipped_gradient(global_model, data, target, args, device))


    # bad clients update their momentums
    for idx in range(len(client_momentums[0])):
        gradient_stack = torch.stack([gradient_list[i][idx] for i in range(num_bad_clients)], dim=0)
        mu = torch.mean(gradient_stack, dim=0)

        if args.attack_type ==  "ipm":
            epsilon = 1  # parameters for IPM attack
            bad_gradient = -epsilon * mu
        elif args.attack_type ==  "alie":
            scale = 4 if args.name_dataset == "mnist" else 1  # parameters for ALIE attack
            std = torch.std(gradient_stack, dim=0)
            bad_gradient = mu - scale * std * z_max
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
    for _ in range(10):
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
            bad_momentum[idx].copy_(  (param_global.data - param.data)/learn_rate )


        for i in range(args.num_bad_clients):
            for idx, param in enumerate(client_momentums[i]):
                param.copy_( bad_momentum[idx] )

    return None

"""
def byzantine_LF(global_model, client_momentums, train_loaders, beta, args, device):
    # Lable Flipping (LF) Attack
    num_bad_clients = args.num_bad_clients
    for i in range(num_bad_clients):
        data, target = next(iter(train_loaders[i]))
        target = 9 - target  # flip the label
        data, target = data.to(device), target.to(device)
        gradient = get_smoothed_and_clipped_gradient(global_model, data, target, args, device)
        for grad_param, momen_param in zip(gradient, client_momentums[i]):
            momen_param.copy_((1-beta)*grad_param + beta*momen_param)

    return None
"""
