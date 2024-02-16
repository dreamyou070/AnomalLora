import torch
anomal_map = torch.randn(1,64*64)
trg = anomal_map.view(1,1,64,64)
print(trg.shape)