import torch

img_attns = torch.tensor([0,0,1,0,])
object_index = [i for i in range(len(img_attns)) if img_attns[i] > 0]
print(object_index)