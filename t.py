import random
from torch import nn
import torch
import einops

query = torch.randn(1,64*64, 320) # B,H*W,C
res = int(query.shape[1] ** 0.5)
query = einops.rearrange(query, 'b (h w) c -> b c h w', h=res, w=res) # B,C,H*W

#pooling_layer = nn.MaxPool2d(kernel_size = (2,2))
pooling_layer = nn.AvgPool2d(kernel_size = (2,2))
out_query = pooling_layer(query)
query = einops.rearrange(out_query, 'b c h w -> b (h w) c') # B,H*W,C

print(out_query.shape) # torch.Size([1, 320, 32, 32])
