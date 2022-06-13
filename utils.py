import os
import numpy as np
import torch
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='recipe', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--num_samples', default=10, type=int)
    parser.add_argument('--min_num_leaf', default=8, type=int)
    parser.add_argument('--ondpp', default=False, type=bool)
    return parser.parse_args()


def psd_matrix_sqrt(A):
    eig_vals, eig_vecs = torch.linalg.eigh(A)
    idx = eig_vals > 1e-15
    return eig_vecs[:,idx] * eig_vals[idx].sqrt()


def load_ndpp_kernel(dataset, ondpp):
    
    if dataset == 'recipe':
        file_path_ondpp = 'recipe_ondpp.torch'
        file_path = 'recipe_ndpp.torch'
    else:
        raise NotImplementedError

    if ondpp:
        saved_model = torch.load(os.path.join("./models/", file_path_ondpp))
    else:
        saved_model = torch.load(os.path.join("./models/", file_path))

    V, B, D = saved_model['V'], saved_model['B'], saved_model['C']
    n = V.shape[0]
    C = D - D.T
    d1 = V.shape[1]

    X = torch.cat((V, B), dim=1)
    W = torch.block_diag(torch.eye(d1), C)
    return X, W


def elementary_polynomial(k, eigen_vals):
    n = len(eigen_vals)
    E_poly = torch.zeros((k + 1, n + 1), dtype=eigen_vals.dtype)
    E_poly[0, :] = 1
    for l in range(1, k + 1):
        for n in range(1, n + 1):
            E_poly[l, n] = E_poly[l, n - 1] + eigen_vals[n - 1] * E_poly[l - 1, n - 1]
    return E_poly


def sample_kdpp_eigen_vecs(k, eig_vals, E_poly, rng=None):
    ind_selected = np.zeros(k, dtype=int)
    for n in range(len(eig_vals), 0, -1):
        rand_nums = rng.rand() if rng else np.random.rand()
        if rand_nums < eig_vals[n - 1] * E_poly[k - 1, n - 1] / E_poly[k, n]:
            k -= 1
            ind_selected[k] = n - 1
            if k == 0:
                break
    return ind_selected

