import os
from PIL import Image
import numpy as np
import torch
from perlin import rand_perlin_2d_np
import imgaug.augmenters as iaa

def load_image(image_path, trg_h, trg_w):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    if trg_h and trg_w:
        image = image.resize((trg_w, trg_h), Image.BICUBIC)
    img = np.array(image, np.uint8)
    return img

def randAugmenter(self):
    aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
    aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                          self.augmenters[aug_ind[1]],
                          self.augmenters[aug_ind[2]]])
    return aug

rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
def augment_image(image, anomaly_source_path):
    aug = randAugmenter()
    perlin_scale = 6
    min_perlin_scale = 0
    anomaly_source_img = load_image(anomaly_source_path, 512,512)

    anomaly_img_augmented = aug(image=anomaly_source_img)
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np((512, 512), (perlin_scalex, perlin_scaley))
    perlin_noise = rot(image=perlin_noise)
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8

    augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)

    no_anomaly = torch.rand(1).numpy()[0]
    if no_anomaly > 0.5:
        image = image.astype(np.float32)
        return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
    else:
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly=0.0
        return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

# [1] Read the image and apply general augmentation
image_path = 'hole_5.png'
image = load_image(image_path, 512,512)
anomaly_source_path = 'test.png'
augmented_image, anomaly_mask, has_anomaly = augment_image(image, anomaly_source_path)

org_img = Image.fromarray(image)
aug_img = Image.fromarray(augmented_image)
org_img.show()
aug_img.show()
