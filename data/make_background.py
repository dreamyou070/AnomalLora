import os
from PIL import Image
import numpy as np
import argparse
def main(args):

    print(f'step 1. img dir')
    image_base_dir = args.image_base_dir
    object_img = os.path.join(image_base_dir, f'{args.trg_cat}/train/good')
    rgb_base_dir = os.path.join(object_img, 'rgb')

    background_base_dir = os.path.join(object_img, f'background')
    os.makedirs(background_base_dir, exist_ok=True)

    object_mask_base_dir = os.path.join(object_img, 'object_mask')
    images = os.listdir(rgb_base_dir)

    for image in images:
        base_img = os.path.join(rgb_base_dir, image)
        base_pil = Image.open(base_img)
        h, w = base_pil.size
        base_np = np.array(base_pil)

        # object mask
        object_mask_dir = os.path.join(object_mask_base_dir, image)
        object_pil = np.array(Image.open(object_mask_dir).resize((w, h)).convert('L'))

        # 3. background
        devide_num = 4
        background_position = np.where(object_pil > 0, 0, 1)
        crop_cord = (0, 0, int(h /devide_num), int(w /devide_num))
        background_np = np.expand_dims(background_position, axis=2) * base_np
        background_pil = Image.fromarray(background_np.astype(np.uint8))
        croped_np = np.array(background_pil.crop(crop_cord))
        back_np = base_np.copy()
        for i in range(devide_num):
            for j in range(devide_num):
                back_np[i * int(h / devide_num):(i + 1) * int(h /devide_num), j * int(w /devide_num):(j + 1) * int(w /devide_num)] = croped_np
        back_pil = Image.fromarray(back_np)
        back_dir = os.path.join(background_base_dir, image)
        back_pil.save(back_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_base_dir', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='carrot')
    args = parser.parse_args()
    main(args)
