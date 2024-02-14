import torch
from torch import nn

classification_map = torch.randn(8, 64*64,1)
classification_map = torch.max(classification_map, dim=0).values.squeeze()
print(classification_map.shape)