import numpy as np
import random
import torch
import einops

query_1 = torch.randn(1,64*64, 320)

fc_layer = torch.nn.Linear(320, 3)
output = fc_layer(query_1)
print(output.shape)
output = einops.rearrange(output, 'b (h w) c -> b c h w', h=64, w=64)
print(output.shape)