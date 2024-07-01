import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-D",
        "--name-dataset",
        type=str,
        default="mnist",
        help="dataset for taining, 'mnist' or 'cifar'  (default: 'mnist')",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="trial.p",
        help="filename to save the result  (default: 'trial.p')",
    )
    parser.add_argument(
        "-n",
        "--num-clients",
        type=int,
        default=100,
        help="number of clients in the federated learning context (default: 100)"
    )
    parser.add_argument(
        "-T",
        "--num-iters",
        type=int,
        default=1000,
        help="number of iterations/rounds to train, i.e., T (default: 1000)"
    )
    parser.add_argument(
        "-r",
        "--num-runs",
        type=int,
        default=1,
        help="number of runs to average on (default: 1)"
    )
    parser.add_argument(
        "-q",
        "--client-selected-rate",
        type=float,
        default=1.0,
        help="the rate of one client selected by the server for agregation, i.e., q",
    )
    parser.add_argument(
        "-lr",
        "--learn-rate",
        type=float,
        default=0.1,
        help="learning rate of the global model, i.e., \eta  (default: 0.1)",
    )
    parser.add_argument(
        "-lrdecay",
        "--lr-decay-final-ratio",
        type=float,
        default=0.1,
        help="learning rate decay: from 1 * ini_learn_rate to lr_decay_final_ratio * ini_learn_rate",
    )
    parser.add_argument(
        "--aggregate-method",
        type=str,
        default="average",
        help="options: 'average', 'median' (default: )",
    )

    # DP-related args
    parser.add_argument(
        "--privacy-type",
        type=str,
        default="no_dp",
        help="options: 'no_dp', 'ldp', 'cdp' (default: )",
    )
    parser.add_argument(
        "-p",
        "--record-sampled-rate",
        type=float,
        default=0.05,
        help="the rate of one record sampled by one client for gradient copmutation, i.e., p",
    )
    
    parser.add_argument(
        "-sigma",
        "--noise-multiplier",
        type=float,
        default=0.02,
        help="noise multiplier, i.e., \sigma (default: )",
    )
    parser.add_argument(
        "-R",
        "--max-record-grad-norm",
        type=float,
        default=10,
        help="clip per-record gradient to this norm for record-level DP (default: )",
    )

    # attack-related args
    parser.add_argument(
        "--attack-type",
        type=str,
        default="no_attack",
        help="options: 'no_attack', 'ipm', 'alie', 'lf', 'mtb' (default: 'no_attack')",
    )
    parser.add_argument(
        "-K",
        "--num-bad-clients",
        type=int,
        default=0,
        help="number of malicious clients who implement Byzantine attacks, i.e., K (default: 0)"
    )

    # defense-related args
    parser.add_argument(
        "--client-clip-type",
        type=str,
        default="center_clip",
        help="options: 'center_clip', or 'direct_clip' ",
    )
    parser.add_argument(
        "-C",
        "--max-client-clip-norm",
        type=float,
        default=5,
        help="clip per-client momentum (or centered clipping) to this norm, i.e., C . If C=-1, then disable clipping",
    )
    parser.add_argument(
        "-beta",
        "--client-momentum",
        type=float,
        default=0,
        help="client momentum for SGD, i.e., beta=1-alpha (default: )",
    )

    # other args
    parser.add_argument(
        "--show-iter-acc",
        action="store_true",
        default=False,
        help="show iteration loss (default: False)",
    )
    parser.add_argument(
        "--use-inefficient-clip",
        action="store_true",
        default=False,
        help="use inefficient clip (only for efficiency evaluation)",
    )
    return parser.parse_args() 



def set_args(name_dataset, aggregate_method, privacy_type, noise_multiplier, attack_type, num_bad_clients,
             num_clients=None, num_iters=None, num_runs=None,
             client_selected_rate=None, max_record_grad_norm=None, 
             client_clip_type=None, max_client_clip_norm=None, client_momentum=None,
             learn_rate=None, lr_decay_final_ratio=None,
             record_sampled_rate=None, 
             print_args=False):

    
    # default values
    args = get_args()
    args.filename = None

    # customized args
    args.name_dataset = name_dataset            # the dataset: mnist or cifar
    args.aggregate_method = aggregate_method
    args.privacy_type = privacy_type
    args.noise_multiplier = noise_multiplier
    args.attack_type = attack_type
    args.num_bad_clients = num_bad_clients

    # (optional) customized args
    if num_clients: args.num_clients = num_clients
    if num_iters: args.num_iters = num_iters
    if num_runs: args.num_runs = num_runs

    if client_selected_rate: args.client_selected_rate = client_selected_rate
    if max_record_grad_norm: args.max_record_grad_norm = max_record_grad_norm

    if client_clip_type: args.client_clip_type = client_clip_type
    if max_client_clip_norm: args.max_client_clip_norm = max_client_clip_norm
    if client_momentum: args.client_momentum = client_momentum

    if learn_rate: args.learn_rate = learn_rate
    if lr_decay_final_ratio: args.lr_decay_final_ratio = lr_decay_final_ratio

    if record_sampled_rate: args.record_sampled_rate = record_sampled_rate
    


    check_args(args, print_args)
    return args



def check_args(args, print_args=True):
    """
    make sure there is no conflicts on the args, and then print the values of args
    """

    if args.aggregate_method not in ['average', 'median', 'sign']:
        raise ValueError("Unrecognized aggregate_method")

    if args.privacy_type not in ['no_dp', 'ldp', 'cdp', 'ddp']:
        raise ValueError("Unrecognized privacy_type")
    
    # when no_dp, we neigher clip nor add noise 
    if args.privacy_type == "no_dp":
        args.max_record_grad_norm = -1
        args.noise_multiplier = 0   

    # check attack_type and num_bad_clients
    if args.attack_type not in ['no_attack', 'ipm', 'alie', 'lf', 'mtb']:
        raise ValueError("Unrecognized attack_type")

    if args.attack_type == 'no_attack':
        args.num_bad_clients = 0
    elif args.num_bad_clients == 0:
        raise ValueError("For this attack_type, the value of num_bad_clients should larger than 0!")
    
    # check lr_decay_final_ratio
    if args.lr_decay_final_ratio < 0.01 or args.lr_decay_final_ratio > 1:
        raise ValueError("lr_decay_final_ratio should in the range [0.01, 1]")

    
    # check client_clip_type
    if args.client_clip_type not in ['center_clip',  'direct_clip', 'element_clip']:
        raise ValueError("Unrecognized client_clip_type")

    if print_args == True:
        print("\n")
        for k, v in vars(args).items():
            print(f"{k} = {v}")

