import torch
import torch.nn as nn
import numpy as np
from PIL import Image

random_map = torch.randn(64*64)
random_map = torch.softmax(random_map, dim=0)
random_map = random_map / random_map.max()
print(random_map.max())