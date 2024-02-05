from scipy.stats import chi2
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal
from random import sample
from scipy.spatial.distance import mahalanobis

def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

t_d = 320
d = 100
# [1] random sample index
idx = torch.tensor(sample(range(0, t_d), d))
#print(idx)
embedding_vectors = torch.randn(100, t_d)
embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
feature = embedding_vectors.numpy()

mu_np = torch.mean(embedding_vectors, dim=0).numpy() # [100 dim, 3136],
conv_np = np.cov(feature, rowvar=False)
conv_inv_np = np.linalg.inv(conv_np)
dist = [mahalanobis(sample, mu_np, conv_np) for sample in embedding_vectors]
print(dist)
print(f'----------------------------------------------------')
mu_torch = torch.mean(embedding_vectors, dim=0)
conv_torch = torch.cov(embedding_vectors.transpose(0, 1))
conv_inv_torch = conv_torch.T
dist_torch = [mahal(sample, mu_torch, conv_inv_torch) for sample in embedding_vectors]
print(dist_torch)