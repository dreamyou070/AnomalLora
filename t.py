import torch.nn as nn
import torch
import einops



positional_encodings = torch.randn(1,64*64, 320)
b_size = 1
pe = positional_encodings.expand(b_size, -1, -1)
print(pe.shape) # torch.Size([1, 4096, 320])