import os
from PIL import Image
import numpy as np
import torch
from perlin import rand_perlin_2d_np
import imgaug.augmenters as iaa

image_path = 'hole_5.png'
anomaly_source_path = 'test.png'

def load_image(image_path, trg_h, trg_w):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    if trg_h and trg_w:
        image = image.resize((trg_w, trg_h), Image.BICUBIC)
    img = np.array(image, np.uint8)
    return img

def augment_image(image, anomaly_source):

    # [1] get anomal mask
    perlin_scale = 6
    min_perlin_scale = 0
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_noise = rand_perlin_2d_np((512,512), (perlin_scalex, perlin_scaley))
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    # [2] synthetic image
    beta = torch.rand(1).numpy()[0] * 0.8
    augmented_image = (1 - perlin_thr) * image + \
                      (perlin_thr) * (beta * image + (1 - beta) * anomaly_source)
    return augmented_image, perlin_thr
def transform_image(image_path, anomaly_source_path):
    img = load_image(image_path, 512,512)
    anomal_img = load_image(anomaly_source_path, 512,512)
    augmented_image, anomaly_mask = augment_image(img, anomal_img)
    return img, augmented_image, anomaly_mask

image, augmented_image, anomaly_mask = transform_image(image_path,anomaly_source_path)


image_path = 'hole_5.png'
image = load_image(image_path, 512,512)
anomaly_source_path = 'test.png'
anomal_image = load_image(anomaly_source_path, 512,512)

augmented_image, anomaly_mask = augment_image(image, anomal_image)
anomal_pil = Image.fromarray((np.squeeze(anomaly_mask, axis=2) * 255).astype(np.uint8)).resize((64,64))
anomal_torch = torch.tensor(np.array(anomal_pil))
anomal_mask = torch.where(anomal_torch > 0, 1, 0)
print(anomal_mask.shape)

