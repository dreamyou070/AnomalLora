import torch.nn as nn
import torch
import einops
num_patches = 64*64
embed_dim = 320

query = torch.randn(1,64*64, 320) # B,H*W,C
res = int(query.shape[1] ** 0.5)
query = einops.rearrange(query, 'b (h w) c -> b c h w', h=res, w=res) # B,C,H*W

conv_filter = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1)
out_query = conv_filter(query)
query = einops.rearrange(out_query, 'b c h w -> b (h w) c') # B,H*W,C
print(query.shape) # torch.Size([1, 320, 64, 64])
# global