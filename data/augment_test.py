import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from PIL import Image, ImageFilter
import cv2
import imgaug.augmenters as iaa
import argparse, os

base_augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                   iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                   iaa.pillike.EnhanceSharpness(),
                   iaa.Sharpen(alpha=(0.0,0.0), ),
                   iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                   iaa.Solarize(0.5, threshold=(32,128)),
                   iaa.Posterize(),
                   iaa.Invert(),
                   iaa.pillike.Autocontrast(),
                   iaa.pillike.Equalize(),
                   iaa.Affine(rotate=(-45, 45))]

def main(args) :

    img_dir = args.img_dir
    base, name = os.path.split(img_dir)
    name, ext = os.path.splitext(name)
    aug_idx = args.aug_idx
    augmenter = base_augmenters[aug_idx]
    print(f' step 1. read img')
    np_img = np.array(Image.open(img_dir))
    print(f' step 2. augment img')
    aug_np_img = augmenter(image=np_img)
    aug_img_pil = Image.fromarray(aug_np_img)
    save_dir = os.path.join(base, f'{name}_aug_{aug_idx}.png')
    aug_img_pil.save(save_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment the object mask')
    parser.add_argument('--img_dir', type=str, default='samples/carrot.png')
    parser.add_argument('--aug_idx', type=int, default=3)
    args = parser.parse_args()
    main(args)