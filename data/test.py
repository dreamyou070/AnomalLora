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
    cov_inv = cov.T
    m_ = torch.matmul(cov_inv, delta)
    m = torch.dot(delta, m_)
    return torch.sqrt(m)

sample_feat = torch.randn(100)
mu = torch.randn(100)
cov = torch.randn(100,100)
dist = mahal(sample_feat.squeeze(), mu.squeeze(), cov)