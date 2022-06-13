import numpy as np
from tqdm import tqdm
from utils import psd_matrix_sqrt
from spectral import spectral_symmetrization
from tree_based_sampling import kdpp_tree_sampling_customized


def kndpp_mcmc(tree, X, W, k, num_walks, rng):
    n = X.shape[0]
    assert k >= 2 and k <= n
    if rng is None:
        rng = np.random.RandomState(None)
    S = rng.permutation(n)[:k]
    num_rejections = []

    for _ in tqdm(range(num_walks), desc='Up-Down Random Walk'):
        T = rng.choice(S, k-2, replace=False)

        Xdown = X[T,:]
        W_cond = W - W @ Xdown.T @ ((Xdown @ W @ Xdown.T).inverse()) @ Xdown @ W
        W_cond_hat = (W_cond + W_cond.T)/2 + spectral_symmetrization((W_cond - W_cond.T)/2)
        What_sqrt = psd_matrix_sqrt(W_cond_hat)
        get_det_L = lambda S : (X[S,:] @ W_cond @ X[S,:].T).det()
        get_det_Lhat = lambda S : (X[S,:] @ W_cond_hat @ X[S,:].T).det()
        cnt = 0
        while(1):
            ab = kdpp_tree_sampling_customized(tree, What_sqrt, X, 2, rng)
            rand_num = rng.rand() if rng else np.random.rand()
            if rand_num < get_det_L(ab) / get_det_Lhat(ab):
                break
            cnt += 1
        num_rejections.append(cnt)
        S = np.union1d(T, ab)
    return S, num_rejections

