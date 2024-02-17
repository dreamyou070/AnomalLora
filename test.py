import torch
import torch.nn as nn
import numpy as np
from PIL import Image

activation_1 = torch.randn(1,320, 64, 64)  # -> 1, 320, 64, 64
activation_2 = torch.randn(1,640, 32, 32)  # -> 1, 640, 32, 32
activation_3 = torch.randn(1,1280, 16, 16) # -> 1, 1280, 16, 16
activation_4 = torch.randn(1,1280, 8, 8)   # -> 1, 1280, 8, 8

text_emb = torch.randn(1,320, 77)
text_emb = torch.randn(1,640, 77)
text_emb = torch.randn(1,1280, 77)
text_emb = torch.randn(1,1280, 77)

activations = [activation_1, activation_2, activation_3, activation_4]

resized_activations = []
for feats in activations:
    feats = nn.functional.interpolate(feats, size=64, mode="bilinear")
    resized_activations.append(feats)

query = torch.randn(8, 64*64, 3520)
key = torch.randn(8, 77, 3520)
attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1],
                                             dtype=query.dtype, device=query.device), query,
                                 key.transpose(-1, -2), beta=0)
attention_probs = attention_scores.softmax(dim=-1)
cls_score, trigger_score = attention_probs[:,:,0], attention_probs[:,:,1]
print(attention_scores.shape)