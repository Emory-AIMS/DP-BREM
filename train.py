import numpy as np
from tqdm.auto import tqdm
import pickle
import time
import os.path 


import torch
import torch.nn as nn
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from data import prepare_data_model
from byzantine_attacks import byzantine_IPM_or_ALIE, byzantine_LF, byzantine_MTB_minmax
from utils import zeros_list_like, clip_list_, get_clipped_gradient
import warnings
warnings.filterwarnings("ignore")



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
            if args.client_clip_type == "element_clip":
                """
                the paper `Privacy-Preserving Robust Federated Learning with Distributed Differential Privacy`
                uses element clip with range [-0.1, 0.1]
                """
                clipped_one = [torch.clamp(param, -0.1, 0.1) for param in clipped_one] 
            elif max_client_clip_norm > 0:
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


def server_aggregate_sign(global_model, client_updates, selected, learn_rate, device):
    """
    "Bridging Differential Privacy and Byzantine-Robustness via Model Aggregation" [IJCAI' 2022]
    Summary: client add local noise (note: the local updates differs from the conventional one), and server aggregate sign
    """
    with torch.no_grad():
        server_grad = zeros_list_like(global_model, device)
        for idx, param in enumerate(server_grad):
            for i in selected:
                param.add_(torch.sign(client_updates[i][idx]))

        # update global model 
        for idx, param in enumerate(global_model.parameters()):
            param = param.data
            server_grad = torch.zeros_like(param).to(device)
            server_grad.add_( 0.004 * param )
            for i in selected:
                server_grad.add_(torch.sign(client_updates[i][idx]))

            param.add_( -learn_rate*0.01 *  server_grad )

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

    # load data, initialize the models
    train_loaders, test_loader, global_model, attacker = prepare_data_model(args, device)

    # initialize "global_momentum" and "client_momentums"
    global_momentum = zeros_list_like(global_model, device)
    client_momentums = [zeros_list_like(global_model, device) for _ in range(num_clients)]

    if args.attack_type == 'mtb' and num_bad_clients > 0:
        attacker.benign_momentums = [zeros_list_like(global_model, device) for _ in range(num_bad_clients)]

    if args.aggregate_method == 'sign':
        client_sign_models = [zeros_list_like(global_model, device) for _ in range(num_clients)]


    # train the model 
    metric_idx = 0
    start_time = time.time()
    for t in tqdm(range(num_iters), position=0, leave=True):

        beta = args.client_momentum

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


        # use learn_rate scheduler which decrease from 1*args.learn_rate to lr_decay_final_ratio*args.learn_rate
        # Note: descrease learn_rate improves the defense to Byzantine attack (fixed learn-rate make it accuracy -> 10%)
        learn_rate = args.learn_rate * ((args.lr_decay_final_ratio - 1)*t/num_iters + 1)

        # select a set of clients (w.p. "client_selected_rate")
        while True:
            selected = np.where(np.random.uniform(0, 1, args.num_clients) <= args.client_selected_rate)[0].astype(int)
            if selected.size:
                break


        # good clients compute gradient and update client momentum
        clients_need_update = good_clients.copy() if beta>0 else np.intersect1d(selected, good_clients)
        for i in clients_need_update:
            data, target = next(iter(train_loaders[i]))
            data, target = data.to(device), target.to(device)
            gradient = get_clipped_gradient(global_model, data, target, args, device, max_record_grad_norm)
            if args.aggregate_method == 'sign':
                # update client_sign_models[i]
                for idx, global_param in enumerate(global_model.parameters()):
                    global_param = global_param.data
                    param = client_sign_models[i][idx]

                    client_momentums[i][idx].copy_(global_param - param)
                    if args.noise_multiplier > 0 and max_record_grad_norm > 0:
                        std = args.noise_multiplier * max_record_grad_norm * learn_rate
                        client_momentums[i][idx].add_(torch.normal(0, std, param.shape).to(device))

                    param.add_(gradient[idx]+0.01*torch.sign(param-global_param), alpha=-learn_rate)

            else:
                if args.privacy_type in ["ldp", "ddp"] and args.noise_multiplier > 0 and max_record_grad_norm > 0:
                    std = args.noise_multiplier * max_record_grad_norm
                    if args.privacy_type == "ddp":
                        std *= 0.3
                    for param in gradient:
                        param.add_(torch.normal(0, std, param.shape).to(device))

                for grad_param, momen_param in zip(gradient, client_momentums[i]):
                    momen_param.copy_((1-beta)*grad_param + beta*momen_param)

        # for mtb attack, update attacker.benign_momentums
        if args.attack_type == 'mtb' and num_bad_clients > 0:
            for i in range(num_bad_clients):
                data, target = next(iter(train_loaders[i]))
                data, target = data.to(device), target.to(device)
                gradient = get_clipped_gradient(global_model, data, target, args, device, max_record_grad_norm)
                for grad_param, momen_param in zip(gradient, attacker.benign_momentums[i]):
                    momen_param.copy_((1-beta)*grad_param + beta*momen_param)


        
        # bad clients update momentum
        if min(selected) < num_bad_clients:  # if at least one bad clients is selected

            if args.attack_type in  ["ipm", "alie"]:
                byzantine_IPM_or_ALIE(global_model, client_momentums, train_loaders, args, device)


            elif args.attack_type ==  "lf":
                byzantine_LF(global_model, client_momentums, attacker, args, device)

            elif args.attack_type == 'mtb':
                byzantine_MTB_minmax(client_momentums, attacker, args)
            
            elif args.attack_type != "no_attack":
                raise ValueError("Something wrong with args.attack_type !")
            

        
        
        # server aggregates selected clients' models (and adds Gaussian noise)
        if args.aggregate_method == "average":
            server_aggregate_average(global_model, global_momentum, client_momentums, selected, learn_rate, max_record_grad_norm, max_client_clip_norm, args, device)
        elif args.aggregate_method == "median":
            server_aggregate_median(global_model, global_momentum, client_momentums, selected, learn_rate, max_record_grad_norm, args, device)
        elif args.aggregate_method == "sign":
            server_aggregate_sign(global_model, client_momentums, selected, learn_rate, device)
        else:
            raise NotImplementedError


        # test the global model on the main task
        if metric_idx < num_points and t == t_list[metric_idx]:
            test_loss[metric_idx], test_acc[metric_idx] = test_model(global_model, test_loader, device)

            if args.show_iter_acc:
                # print the process
                print("t={},  test_loss={:.3f},  test_acc={:.1f}%".format(t, test_loss[metric_idx], test_acc[metric_idx]*100))

            metric_idx += 1

    print(f"Training time: {int(time.time() - start_time)} seconds")

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
    np.set_printoptions(precision=4)
    print("test_acc = ", result["test_acc"])
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
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