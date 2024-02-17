from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np
from rembg import remove
def remove_bg(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

def main(args):

    print(f'step 1. prepare images')
    source_folder = os.path.join(args.source_folder, f'{args.bench_mark}/{args.obj_name}/train/good')
    back_rm_object_mask_folder = os.path.join(source_folder, f'back_rm_object_mask')
    mask_images = os.listdir(back_rm_object_mask_folder)

    for mask_img in mask_images:
        mask_path = os.path.join(back_rm_object_mask_folder, mask_img)
        image = Image.open(mask_path)
        np_img = np.array(image)
        h, w = image.size
        for h_index in range(h):
            for w_index in range(w):
                if h_index < int(h / 10) or h_index > int(h * 9 / 10):
                    np_img[h_index, w_index] = 0
        for w_index in range(w):
            for h_index in range(h):
                if w_index < int(w / 90) or w_index > int(w * 89 / 90):
                    np_img[h_index, w_index] = 0
        image = Image.fromarray(np_img)
        image.save(mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--source_folder", help="Path to the input image",
                        default="/home/dreamyou070/MyData/anomaly_detection")
    parser.add_argument("--bench_mark", default="MVTec3D-AD", type=str, help="MVTec3D-AD or MVTecAD")
    parser.add_argument("--obj_name", default="carrot", type=str)
    args = parser.parse_args()
    main(args)
