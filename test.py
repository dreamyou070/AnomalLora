import torch


attn_loss = torch.randn(4)
attn_loss = attn_loss.mean()
print(attn_loss)