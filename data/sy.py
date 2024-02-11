import torch
import numpy as np
from perlin import rand_perlin_2d_np
import cv2
import skimage
import skimage.exposure
from PIL import Image

perlin_max_scale = 8
kernel_size = 5
def make_random_mask(height, width) -> np.ndarray:
    perlin_scale = perlin_max_scale
    min_perlin_scale = 4
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    noise = rand_perlin_2d_np((height, width), (perlin_scalex, perlin_scaley))

    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_pil = Image.fromarray(mask).convert('L')
    mask_np = np.array(mask_pil) / 255  # height, width, [0,1]
    return mask_np, mask_pil

mask_np, mask_pil = make_random_mask(512,512)
mask_pil.show()