from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np
from rembg import remove
import shutil
def remove_bg(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

def main(args):

    print(f'step 1. prepare model')
    model_type = "vit_h"
    path_to_checkpoint= r'/home/dreamyou070/pretrained_stable_diffusion/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=path_to_checkpoint)
    predictor = SamPredictor(sam)

    print(f'step 2. prepare images')
    source_folder = os.path.join(args.source_folder, f'{args.bench_mark}/{args.obj_name}/test')
    defect_folders = os.listdir(source_folder)
    for defect_folder in defect_folders:

        defect_folder_dir = os.path.join(source_folder, defect_folder)

        rgb_folder = os.path.join(defect_folder_dir, f'rgb')
        rgb_origin_folder = os.path.join(defect_folder_dir, f'rgb_origin')
        os.makedirs(rgb_origin_folder, exist_ok=True)
        rgb_bgrm_folder = os.path.join(defect_folder_dir, f'rgb_bgrm')
        os.makedirs(rgb_bgrm_folder, exist_ok=True)

        rgb_imgs = os.listdir(rgb_folder)

        for rgb_img in rgb_imgs :

            # [1] copy
            rgb_img_path = os.path.join(rgb_folder, rgb_img)
            rgb_org_path = os.path.join(rgb_origin_folder, rgb_img)
            shutil.copy(rgb_img_path, rgb_org_path)

            # [2] remove background
            rmbg_path = os.path.join(rgb_bgrm_folder, rgb_img)
            remove_bg(rgb_org_path, rmbg_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--source_folder", help="Path to the input image",
                        default="/home/dreamyou070/MyData/anomaly_detection")
    parser.add_argument("--bench_mark", default="MVTec3D-AD", type=str, help="MVTec3D-AD or MVTecAD")
    parser.add_argument("--obj_name", default="carrot", type=str)
    args = parser.parse_args()
    main(args)
