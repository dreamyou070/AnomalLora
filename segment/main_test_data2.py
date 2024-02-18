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
        back_rm_object_mask_folder = os.path.join(defect_folder_dir, f'back_rm_object_mask')
        os.makedirs(back_rm_object_mask_folder, exist_ok=True)
        imgs = os.listdir(rgb_folder)

        for img in imgs :

            # [1] read img
            input_path = os.path.join(rgb_folder, img)
            rmgb_pil = Image.open(input_path).convert("RGB")
            org_h, org_w = rmgb_pil.size

            # [2] object segment
            rmbg_np = np.array(rmgb_pil)
            predictor.set_image(rmbg_np)
            h, w, c = rmbg_np.shape
            input_point = np.array([[0, 0]])
            input_label = np.array([1])
            masks, scores, logits = predictor.predict(point_coords=input_point,
                                                      point_labels=input_label,
                                                      multimask_output=True, )
            for i, (mask, score) in enumerate(zip(masks, scores)):
                if i == 1:
                    np_mask = (mask * 1)
                    np_mask = np.where(np_mask == 1, 0, 1) * 255
                    sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                    sam_result_pil = sam_result_pil.resize((org_h, org_w))
                    sam_result_pil.save(os.path.join(back_rm_object_mask_folder, img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--source_folder", help="Path to the input image",
                        default="/home/dreamyou070/MyData/anomaly_detection")
    parser.add_argument("--bench_mark", default="MVTec3D-AD", type=str, help="MVTec3D-AD or MVTecAD")
    parser.add_argument("--obj_name", default="carrot", type=str)
    args = parser.parse_args()
    main(args)
