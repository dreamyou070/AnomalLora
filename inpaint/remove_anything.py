import torch
from PIL import Image
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",)
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latest_coords = args.point_coords

    data_base_dir = '/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD'
    object = args.object
    object_base_dir = os.path.join(data_base_dir, object)
    train_dir = os.path.join(object_base_dir, 'train')
    train_good_dir = os.path.join(train_dir, 'good')
    train_good_rgb_dir = os.path.join(train_good_dir, 'rgb')
    images = os.listdir(train_good_rgb_dir)

    for image in images:
        img_path = os.path.join(train_good_rgb_dir, image)
        img = load_img_to_array(img_path)
        h, w = img.shape[:2]
        masks, _, _ = predict_masks_with_sam(img,[latest_coords],args.point_labels,
                                             model_type=args.sam_model_type,
                                             ckpt_p=args.sam_ckpt,device=device,)
        masks = masks.astype(np.uint8) * 255 # np array

        # dilate mask to avoid unmasked edge effect
        if args.dilate_kernel_size is not None:
            masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

        # visualize the segmentation results
        img_stem = Path(img_path).stem
        out_dir = Path(args.output_dir) / img_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, mask in enumerate(masks):
            # path to the results
            mask_p = out_dir / f"mask_{idx}.png"
            img_points_p = out_dir / f"with_points.png"
            img_mask_p = out_dir / f"with_{Path(mask_p).name}"

            # -------------------------------------------- saving mask -------------------------------------------- #
            Image.fromarray(mask.astype(np.uint8)).resize((h, w)).save(mask_p)


            # save the pointed and masked image
            dpi = plt.rcParams['figure.dpi']
            height, width = img.shape[:2]
            plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
            plt.imshow(img)
            plt.axis('off')
            show_points(plt.gca(), [latest_coords], args.point_labels,
                        size=(width*0.04)**2)
            plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
            show_mask(plt.gca(), mask, random_color=False)
            plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
            plt.close()

            # -------------------------------------------- inpaint the masked image ---------------------------------- #
            #
            for idx, mask in enumerate(masks):
                mask_p = out_dir / f"mask_{idx}.png"
                img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
                img_inpainted = inpaint_img_with_lama(
                    img, mask, args.lama_config, args.lama_ckpt, device=device)
                save_array_to_img(img_inpainted, img_inpainted_p)