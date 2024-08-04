# DP-BREM: Differentially-Private and Byzantine-Robust Federated Learning with Client Momentum
This is the code of our work [DP-BREM](https://arxiv.org/abs/2306.12608) [USENIX Security' 25]

## Two ways to run this code

1. Directly run `main.py` with the required parameters, see `get_args()` in `arg.py` for more details.
2. Use `set_args()` in `arg.py` to get the object of args, then pass it into `train_one_run_and_write_to_file(args, filename)` in `run.py`. If want to process multiple jobs, pass a list of args to `sequntial_train_and_write_file(args_list, filename_list).`
