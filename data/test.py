import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import random
import glob
from data.perlin import rand_perlin_2d_np
from PIL import Image
from torchvision import transforms
from numpy.random import default_rng
import cv2
import skimage.exposure


def load_image(image_path, trg_h, trg_w, type='RGB'):
    image = Image.open(image_path)
    if type == 'RGB':
        if not image.mode == "RGB":
            image = image.convert("RGB")
    elif type == 'L':
        if not image.mode == "L":
            image = image.convert("L")
    if trg_h and trg_w:
        image = image.resize((trg_w, trg_h), Image.BICUBIC)
    img = np.array(image, np.uint8)
    return img


def make_random_gaussian_mask():

    end_num = 512

    # [1] back
    x = np.arange(0, end_num, 1, float)
    y = np.arange(0, end_num, 1, float)[:, np.newaxis]

    # [2] center
    x_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()
    y_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()

    # [3] sigma
    sigma = torch.randint(20, 50, (1,)).item()

    # [4] make kernel
    result = np.exp(-4 * np.log(2) * ((x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2)  # 0 ~ 1
    return result


# [1] base
img_path = '001_org.png'
img = load_image(img_path, 512,512) # np.array,
dtype = img.dtype
# [2] background
background_dir = '001_org.png'
background_img = load_image(background_dir, 512,512)
# [3] object mask
object_mask_dir = '001_org.png'
object_img = load_image(object_mask_dir, 64,64, type='L')
object_mask_np = np.where((np.array(object_img, np.uint8) / 255) == 0, 0, 1)
object_mask = torch.tensor(object_mask_np) # shape = [64,64], 0 = background, 1 = object

object_img_aug = load_image(object_mask_dir, 512,512, type='L') # [512,512]
object_mask_np_aug = np.where((np.array(object_img_aug) / 255) == 0, 0, 1)                              # [512,512]
# [3-1] anomal mask
while True:

    hold_mask_np = make_random_gaussian_mask()
    hold_mask_np = hold_mask_np * object_mask_np_aug  # 1 = hole, 0 = normal
    if hold_mask_np.sum() > 0:
        break
print(hold_mask_np.sum())
