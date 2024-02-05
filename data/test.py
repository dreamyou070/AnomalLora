import torch
from data.perlin import rand_perlin_2d_np
import numpy as np
from PIL import Image

perlin_scale = 6
min_perlin_scale = 0
rand_1 = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
rand_2 = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
perlin_scalex, perlin_scaley = 2 ** (rand_1), 2** (rand_2)
perlin_noise = rand_perlin_2d_np((64*64, 320), (perlin_scalex, perlin_scaley))
perlin_noise = torch.tensor(perlin_noise)
print(perlin_noise)