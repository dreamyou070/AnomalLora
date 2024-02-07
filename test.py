import torch

loss_1 = torch.randn(1)
loss_2 = torch.randn(1)
loss_3 = torch.randn(1)
print(loss_1, loss_2, loss_3)
loss = loss_1 + loss_2 + loss_3
print(loss)