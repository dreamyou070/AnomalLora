import os, json,torch

feat = torch.randn(64*64)
anomal_feat_list = [feat,feat]
student_anomal_feat = torch.stack(anomal_feat_list, dim=0)
print(student_anomal_feat.shape)