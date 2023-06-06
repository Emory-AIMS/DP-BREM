import numpy as np
from tqdm.auto import tqdm
import pickle


import torch
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from data import prepare_data_model
from byzantine_attacks import byzantine_IPM_or_ALIE, byzantine_LF
from utils import zeros_list_like, clip_list_, get_smoothed_and_clipped_gradient


def backdoor_update(global_model, client_momentums, backdoor, args, device):
    """
    How backdoor clients update. We only use data poisoning attack with more local training steps 
    but without model replacement.
    """
    criterion = nn.CrossEntropyLoss()
    model = backdoor.model
    optimizer = backdoor.optimizer

    # sync the global model to the backdoor model
    model.load_state_dict(global_model.state_dict())


    # train the backdoor model with 5 steps
    model.train()
    for _ in range(5):

        data_benign, target_benign = next(iter(backdoor.train_loader))
        data_poison, target_poison = next(iter(backdoor.train_loader_poison))

        data, target = torch.cat((data_benign, data_poison)), torch.cat((target_benign, target_poison))
        data, target = data.to(device), target.to(device)

        # Remark: use model.zero_grad() instead of optimizer.zero_grad(), 
        # otherwise the GUP usage will keep increasing (because we use GradSampleModule)
        model.zero_grad()  
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    

    with torch.no_grad():

        # replace the backdoor momentum by the difference to the global model
        bad_momentum = zeros_list_like(global_model, device)
        for idx, (param_global, param) in enumerate(zip(global_model.parameters(), model.parameters())):
            bad_momentum[idx].copy_(  param_global.data - param.data )


        # since the expected number of selected bad clients is "num_expected_bad", 
        # we assume that they can coordinate with each other and divide the bad gradient
        for i in range(args.num_bad_clients):
            for idx, param in enumerate(client_momentums[i]):
                param.copy_( bad_momentum[idx] )

    return None


def server_aggregate_median(global_model, global_momentum, client_momentums, selected, learn_rate, max_record_grad_norm, args, device):
    """
    This function computes the Coordinate-wise median of clients' gradient, adds Gaussian noise and update global model
    Note: in this aggregation method, we neigher do client_level_clipping nor implement momentum (i.e., only aggregate gradient)
    """
    with torch.no_grad():
        median = zeros_list_like(global_model, device)
        for idx, param in enumerate(median):
            stacked_param = torch.stack([client_momentums[i][idx] for i in selected], dim=0)
            param.copy_(torch.quantile(stacked_param, 0.5, dim=0))  # compute the median

        # add Gaussian noise if necessary
        if args.privacy_type == "cdp" and args.noise_multiplier > 0 and max_record_grad_norm > 0:
            std = args.noise_multiplier * max_record_grad_norm
            for param in median:
                param.add_(torch.normal(0, std, param.shape).to(device))

        # update global_momentum
        for idx, param in enumerate(global_momentum):
            param.copy_(median[idx])
        
        # update global model 
        for idx, param in enumerate(global_model.parameters()):
            param.data.add_( -learn_rate *  global_momentum[idx] )

    return None



def server_aggregate_average(global_model, global_momentum, client_momentums, selected, learn_rate, max_record_grad_norm, max_client_clip_norm, args, device):
    """
    This function compute average of clients' momentum, adds Gaussian noise and update global model
    """
    with torch.no_grad():

    
        # initialize the difference for client and the sum 
        clipped_sum = zeros_list_like(global_model, device)
        clipped_one = zeros_list_like(global_model, device)
        for i in selected:

            # determine the clipping objective
            for idx, param in enumerate(clipped_one):
                if args.client_clip_type == "center_clip":
                    param.copy_(client_momentums[i][idx] - global_momentum[idx])
                else:
                    param.copy_(client_momentums[i][idx])
                    

            # clipping (if necessary)
            if max_client_clip_norm > 0:
                clip_list_(clipped_one, max_client_clip_norm)

            # add the clipped one to the sum
            for idx, param in enumerate(clipped_sum):
                param.add_(clipped_one[idx])

        # add Gaussian noise if necessary
        if args.privacy_type == "cdp" and args.noise_multiplier > 0 and max_record_grad_norm > 0:
            std = args.noise_multiplier * max_record_grad_norm
            for param in clipped_sum:
                param.add_(torch.normal(0, std, param.shape).to(device))

        
        # update global_momentum
        for idx, param in enumerate(global_momentum):
            if args.client_clip_type == "center_clip":
                param.add_(clipped_sum[idx]/len(selected))
            else:
                param.copy_(clipped_sum[idx]/len(selected))
                

        # update global model 
        for idx, param in enumerate(global_model.parameters()):
            param.data.add_( -learn_rate *  global_momentum[idx] )

    return None



def test_model(model, test_loader, device):
    """This function tests the model on test data and returns test loss and test accuracy """
    loss_func = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():

        for cnt, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            # mean per batch
            test_loss += loss_func(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    test_loss /= cnt + 1

    return test_loss, accuracy
            


def test_model_duplicate(model, test_loader, device):
    """
    When testing cifar backdoor task, we need to duplicate the testing examples (but with crop and random flip of the image) to get enought testing examples
    """
    loss_func = nn.CrossEntropyLoss()
    data, target = [], []
    for _ in range( int(1000/len(test_loader)) ):
        data_temp, target_temp = next(iter(test_loader))
        data.append(data_temp)
        target.append(target_temp)
    data, target = torch.cat(data), torch.cat(target)
    data, target = data.to(device), target.to(device)
    output = model(data)
    test_loss = loss_func(output, target).item()
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(pred)
    return test_loss, accuracy




def train_one_run(args, r=None, return_dict=None):
    """
    This function train the model with one run. The evaluation results (e.g., loss/accuracy) are returned via "return_dict" or directly via "result"
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_list = np.linspace(1, args.num_iters-1, num=min(50, args.num_iters-1), dtype=int)

    num_points = len(t_list)
    num_iters = args.num_iters
    num_bad_clients = args.num_bad_clients
    num_clients = args.num_clients
    good_clients = np.arange(num_bad_clients, num_clients)

    test_loss, test_acc = (
        np.zeros(num_points),
        np.zeros(num_points),
    )
    backdoor_test_loss, backdoor_test_acc = np.zeros(num_points), np.zeros(num_points)

    # load data, initialize the models
    train_loaders, test_loader, global_model, attacker = prepare_data_model(args, device)

    # initialize "global_momentum" and "client_momentums"
    global_momentum = zeros_list_like(global_model, device)
    client_momentums = [zeros_list_like(global_model, device) for _ in range(num_clients)]


    # train the model 
    idx = 0
    for t in tqdm(range(num_iters), position=0, leave=True):

        beta = args.client_momentum if t>0 else 0

        if args.max_record_grad_norm < 0:
            max_record_grad_norm = -1
        else:
            # decreasing from 1*max_record_grad_norm to 0.3*args.max_record_grad_norm
            max_record_grad_norm = args.max_record_grad_norm * (-0.7*min(t/num_iters, 1) + 1)  

        if args.max_client_clip_norm < 0:
            max_client_clip_norm = -1
        else:
            # decreasing from 1*max_client_clip_norm to 0.3*max_client_clip_norm
            max_client_clip_norm = args.max_client_clip_norm * (-0.7*min(t/num_iters, 1) + 1) 


        # use learn_rate scheduler which decrease from 1*args.learn_rate to 0.1*args.learn_rate
        learn_rate = args.learn_rate * (-0.9*t/num_iters + 1)  

        # select a set of clients (w.p. "client_selected_rate")
        while True:
            selected = np.where(np.random.uniform(0, 1, args.num_clients) <= args.client_selected_rate)[0].astype(int)
            if selected.size:
                break


        # good clients compute gradient and update client momentum
        clients_need_update = good_clients.copy() if beta>0 or t==0 else np.intersect1d(selected, good_clients)
        for i in clients_need_update:
            data, target = next(iter(train_loaders[i]))
            data, target = data.to(device), target.to(device)
            gradient = get_smoothed_and_clipped_gradient(global_model, data, target, args, device, max_record_grad_norm)
            if args.privacy_type == "ldp" and args.noise_multiplier > 0 and max_record_grad_norm > 0:
                std = args.noise_multiplier * max_record_grad_norm
                for param in gradient:
                    param.add_(torch.normal(0, std, param.shape).to(device))

            for grad_param, momen_param in zip(gradient, client_momentums[i]):
                momen_param.copy_((1-beta)*grad_param + beta*momen_param)


        
        # bad clients update momentum
        if min(selected) < num_bad_clients:  # if at least one bad clients is selected
            if args.attack_type ==  "backdoor":
                # bad clients train one backdoor model and update their momentums
                backdoor_update(global_model, client_momentums, attacker, args, device)

            elif args.attack_type in  ["ipm", "alie"]:
                byzantine_IPM_or_ALIE(global_model, client_momentums, train_loaders, args, device)


            elif args.attack_type ==  "lf":
                byzantine_LF(global_model, client_momentums, attacker, learn_rate, args, device)
            
            elif args.attack_type != "no_attack":
                raise ValueError("Something wrong with args.attack_type !")
            

        
        
        # server aggregates selected clients' models (and adds Gaussian noise)
        if args.aggregate_method == "average":
            server_aggregate_average(global_model, global_momentum, client_momentums, selected, learn_rate, max_record_grad_norm, max_client_clip_norm, args, device)
        elif args.aggregate_method == "median":
            server_aggregate_median(global_model, global_momentum, client_momentums, selected, learn_rate, max_record_grad_norm, args, device)
        else:
            raise NotImplementedError


        # test the global model on the main task and backdoor subtask (if applicable)
        if idx < num_points and t == t_list[idx]:
            test_loss[idx], test_acc[idx] = test_model(global_model, test_loader, device)
            if args.attack_type ==  "backdoor":
                # only when testing backdoor subtask of cifar, we need to duplicate test data
                if args.name_dataset == "cifar":
                    backdoor_test_loss[idx], backdoor_test_acc[idx] = test_model_duplicate(
                            global_model, attacker.test_loader, device)
                else:
                    backdoor_test_loss[idx], backdoor_test_acc[idx] = test_model(
                        global_model, attacker.test_loader, device)

            if args.show_iter_acc:
                # print the process
                print("t={},  test_loss={:.3f},  test_acc={:.1f}%".format(t, test_loss[idx], test_acc[idx]*100))
                if args.attack_type ==  "backdoor":
                    print("   backdoor_test_loss={:.3f},  backdoor_test_acc={:.1f}%".format(backdoor_test_loss[idx], backdoor_test_acc[idx]*100))


            idx += 1

    # determine how to return the result
    if r != None and return_dict != None:
        return_dict[r] = (t_list, test_loss, test_acc)
        return

    else:
        result = {}
        result["t_list"], result["test_loss"],  result["test_acc"] = t_list, test_loss, test_acc
        return result






def train_multi_runs(args):
    """
    train multiple runs to get the averaged curve (via torch.multiprocessing)
    """

    if args.num_runs == 1:
        result = train_one_run(args)

    else:
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for r in range(args.num_runs):
            proc = mp.Process(target=train_one_run, args=(args, r, return_dict))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

        

        # save the averaged result w.r.t. runs (result is a dictionary)
        output = np.array(return_dict.values()).mean(axis=0)
        result = {}
        result["t_list"], result["test_loss"], result["test_acc"] = output[0], output[1], output[2]

    
    return result


def train_one_run_and_write_to_file(args, filename):
    result = train_multi_runs(args)
    print("test_acc = ", result["test_acc"])
    pickle.dump([args, result], open(filename, "wb"))
    print("Completed filename = ", filename)


def sequntial_train_and_write_file(args_list, filename_list):
    for args, filename in zip(args_list, filename_list):
        train_one_run_and_write_to_file(args, filename)


def multi_process_train_and_write_file(args_list, filename_list):
    procs = []
    for args, filename in zip(args_list, filename_list):
        proc = mp.Process(target=train_one_run_and_write_to_file, args=(args, filename))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()