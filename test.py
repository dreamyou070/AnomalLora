import torch
from torch import nn

atorch = torch.randn(64)
a_list = [atorch, atorch, atorch]

b = torch.stack(a_list, dim=0).mean(dim=0)
print(b.shape)