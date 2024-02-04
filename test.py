import torch

normal_feats = []
normal_feat = torch.randn(30)
normal_feats.append(normal_feat.unsqueeze(0))
normal_feats = torch.cat(normal_feats, dim=0)
print(normal_feats.shape)