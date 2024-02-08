import torch

a = torch.randn(8,64*64)
a_ = a.mean(dim=0)
print(a_.shape)