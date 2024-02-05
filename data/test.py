import torch
from data.perlin import rand_perlin_2d_np
import numpy as np
from PIL import Image

org_dir = 'org.png'
org_h, org_w = Image.open(org_dir).size

perlin_scale = 6
min_perlin_scale = 0

rand_1 = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
perlin_scalex = 2 ** (rand_1)

perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
perlin_noise = rand_perlin_2d_np((org_h, org_w), (perlin_scalex, perlin_scaley))
threshold = 0.2
perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))

perlin_thr_pil = Image.fromarray((perlin_thr * 255).astype(np.uint8))
perlin_thr_pil.save(f"perlin_thr_{perlin_scalex}_{perlin_scaley}.png")

org_np = np.array(Image.open(org_dir))
anomal_source = np.ones_like(org_np) * 255
perlin_thr = np.expand_dims(perlin_thr, axis=2)
anomaly_img = (1-perlin_thr) * org_np + (perlin_thr) * anomal_source
anomaly_img_pil = Image.fromarray(anomaly_img.astype(np.uint8))
anomaly_img_pil.save("org_anomal.png")