import numpy as np
import torch

dists = [1,2,3]
max_dist = torch.tensor(dists).max()
print(max_dist)
