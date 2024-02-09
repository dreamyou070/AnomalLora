import cv2
import glob
import os
import numpy as np
import torch
from perlin import rand_perlin_2d_np
from numpy.random import default_rng
import skimage
from PIL import Image

def maie_noise (rand_x, rand_y) :
    #perlin_scale = 6
    #min_perlin_scale = 0
    #rand_x = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
    #rand_y = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
    perlin_scalex = 2 ** rand_x
    perlin_scaley = 2 ** rand_y

    # 1. make perlin noise
    # 주파수 : 클 수록 조밀한 노이즈
    # 옥타브 : 노이즈 중첩. 옥타브 클 수록 높은 주파수와 낮은 주파수가 겹쳐진다.
    noise = rand_perlin_2d_np(shape = (512,512),
                              res = (perlin_scalex, perlin_scaley))
    # Gaussian Blur 을 해줌으로써, 노이즈를 부드럽게 한다.
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # 3. thresholding ( check only brighter part )
    thresh = cv2.threshold(src = stretch, thresh = 175, maxval = 255, type = cv2.THRESH_BINARY)[1] # if more than 175, then 255, else 0
    thresh_pil = Image.fromarray(thresh).convert('L')
    #thresh_pil.show()

    #kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (9, 9))
    #kernel_pil = Image.fromarray(kernel*255).convert('L')
    #kernel_pil.show()


    #mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # more black
    #print(mask)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # more white
    #mask_pil = Image.fromarray(mask).convert('L')
    return thresh_pil
for i in range(100):

    rand_x = torch.randint(0, 7, (1,)).numpy()[0]
    rand_y = torch.randint(0, 7, (1,)).numpy()[0]
    mask_pil = maie_noise(7, 7)
    mask_pil.save(f'sample/{i}_{rand_x}_{rand_y}.png')
