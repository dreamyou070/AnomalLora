import torch.nn as nn
import torch
num_patches = 64*64
embed_dim = 320

pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))



positional_embedder = PositionalEmbedding(d_model=embed_dim,
                                          max_len=num_patches)
x = torch.randn(1,64*64, 320)
x = positional_embedder(x)
print(x.shape)