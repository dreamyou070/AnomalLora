import torch

loss_l2 = torch.nn.modules.loss.MSELoss(reduction='none')
a = torch.randn(8, 1, 3,3)
b = torch.randn(8, 1, 3,3)
loss = loss_l2(a, b)
org = 0
loss = org + loss
print(loss.shape)