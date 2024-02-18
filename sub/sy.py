import torch
normal_cls_loss_list = [torch.randn(64),torch.randn(64)]
normal_cls_loss = torch.stack(normal_cls_loss_list, dim=0).mean()
print(normal_cls_loss.shape)