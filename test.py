import torch

object_attention_mask = torch.randn(1,64*64)
object_attention_mask = object_attention_mask.unsqueeze(-1).repeat(1,1,77)
print(object_attention_mask.shape)