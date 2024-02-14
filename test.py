
import torch
a_ist = [torch.randn(1),torch.randn(1)]
a_mean = torch.tensor(a_ist).mean()
print(a_mean)