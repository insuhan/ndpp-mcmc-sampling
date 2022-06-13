import time
import numpy as np
import torch
from tqdm import tqdm

from tree_based_sampling import construct_tree, construct_tree_fat_leaves
from kndpp import kndpp_mcmc
from utils import load_ndpp_kernel, get_arguments


def TEST_kndpp_real_dataset(dataset='uk', k=10, random_state=1, ondpp=False, min_num_leaf=8, num_samples=10):
    
    rng = np.random.RandomState(random_state)
    torch.random.manual_seed(random_state if random_state else rng.randint(99))

    X, W = load_ndpp_kernel(dataset, ondpp=ondpp)

    n, d = X.shape

    # Preprocessing - tree construction
    tic = time.time()
    print("[MCMC] Tree construction")
    if n >= 1e5:
        tree = construct_tree_fat_leaves(np.arange(n), X.T, min_num_leaf)
    else:
        tree = construct_tree(np.arange(n), X.T)
    time_tree_mcmc = time.time() - tic
    print(f"[MCMC] tree construction time: {time_tree_mcmc:.5f} sec")

    # Set the mixing time to k^2
    num_walks = k**2
    for i in range(num_samples):
        tic = time.time()
        sample, num_rejects = kndpp_mcmc(tree, X, W, k, num_walks, rng)
        time_sample = time.time() - tic
        print(f"[MCMC] sampling time : {time_sample:.5f} sec")
        print(f"[MCMC] num_rejections: {np.mean(num_rejects)}")


if __name__ == "__main__":

    print("k-NDPP MCMC Experiment")
    args = get_arguments()
    for name_, value_ in args.__dict__.items():
        print(f"{name_:<20} : {value_}")
        
    TEST_kndpp_real_dataset(args.dataset, args.k, args.seed, args.min_num_leaf, args.num_samples)
