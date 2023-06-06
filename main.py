import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from train import train_multi_runs
from args import get_args, check_args






if __name__ == "__main__":

    # avoid error of Jupyter's IPython console
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"


    args = get_args()
    check_args(args)
    result = train_multi_runs(args)

    
    # write the args and result to a file. To load: args, result = pickle.load(open("trial.p","rb"))
    pickle.dump([args, result], open(args.filename, "wb"))
    print("Complete!")


