import importlib, argparse, math, sys, random, time, json
import torch
from data.mvtec_sy import MVTecDRAEMTrainDataset
from model.tokenizer import load_tokenizer
import numpy as np
from model.unet import unet_passing_argument
from utils.attention_control2 import passing_argument
import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def main(args):

    print(f'\n step 2. dataset')
    obj_name = args.obj_name
    root_dir = f'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/{obj_name}/train_1/good/rgb'
    num_images = len(os.listdir(root_dir))
    print(f'num_images: {num_images}')
    args.anomaly_source_path = f'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/anomal_source_{obj_name}'
    tokenizer = load_tokenizer(args)
    dataset = MVTecDRAEMTrainDataset(root_dir=root_dir,
                                     anomaly_source_path=args.anomaly_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer,
                                     caption='cable',
                                     use_perlin=True,
                                     num_repeat=1,
                                     anomal_only_on_object=True,
                                     anomal_training=True,
                                     latent_res=64,
                                     perlin_max_scale=args.perlin_max_scale,
                                     kernel_size=args.kernel_size,
                                     beta_scale_factor=args.beta_scale_factor)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    beta_scale_factor = args.beta_scale_factor
    check_base_dir = f'/home/dreamyou070/data_check/{obj_name}/beta_scale_factor_{beta_scale_factor}_self_aug'
    os.makedirs(check_base_dir, exist_ok=True)

    for sample in train_dataloader :

        name = sample['image_name'][0]
        image_name = sample['anomal_name'][0]

        image = sample['image'].squeeze() # [3,512,512]
        np_img = np.array(((image + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_image = Image.fromarray(np_img)
        pil_image.save(os.path.join(check_base_dir, f'{image_name}.png'))

        object_mask = sample['object_mask']
        np_object_mask = object_mask.squeeze().numpy()
        pil_object_mask = Image.fromarray((np_object_mask * 255).astype(np.uint8))
        pil_object_mask.save(os.path.join(check_base_dir, f'{image_name}_object_mask.png'))

        augmented_image = sample['augmented_image'].squeeze()
        np_augmented_image = np.array(((augmented_image + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_augmented_image = Image.fromarray(np_augmented_image)
        pil_augmented_image.save(os.path.join(check_base_dir, f'{image_name}_augmented_image.png'))

        anomaly_mask = sample['anomaly_mask']
        np_anomaly_mask = anomaly_mask.squeeze().numpy()
        pil_anomaly_mask = (np_anomaly_mask * 255).astype(np.uint8)
        pil_anomaly_mask = Image.fromarray(pil_anomaly_mask)
        pil_anomaly_mask.save(os.path.join(check_base_dir, f'{image_name}_anomaly_mask.png'))

        masked_image = sample['masked_image'].squeeze()
        np_masked_image = np.array(((masked_image + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_masked_image = Image.fromarray(np_masked_image)
        pil_masked_image.save(os.path.join(check_base_dir, f'{image_name}_masked_image.png'))

        masked_image_mask = sample['masked_image_mask']
        np_masked_image_mask = masked_image_mask.squeeze().numpy()
        pil_masked_image_mask = (np_masked_image_mask * 255).astype(np.uint8)
        pil_masked_image_mask = Image.fromarray(pil_masked_image_mask)
        pil_masked_image_mask.save(os.path.join(check_base_dir, f'{image_name}_masked_image_mask.png'))

        self_aug_img = sample['self_augmented_image'].squeeze()
        np_self_aug_img = np.array(((self_aug_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_self_aug_img = Image.fromarray(np_self_aug_img)
        pil_self_aug_img.save(os.path.join(check_base_dir, f'{image_name}_self_aug_img.png'))

        self_aug_img_mask = sample['self_augmented_mask']
        np_self_aug_img_mask = self_aug_img_mask.squeeze().numpy()
        pil_self_aug_img_mask = (np_self_aug_img_mask * 255).astype(np.uint8)
        pil_self_aug_img_mask = Image.fromarray(pil_self_aug_img_mask)
        pil_self_aug_img_mask.save(os.path.join(check_base_dir, f'{image_name}_self_aug_img_mask.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--wandb_api_key', type=str, default='output')
    parser.add_argument('--wandb_project_name', type=str, default='bagel')
    # step 2. dataset
    parser.add_argument('--data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='cable_gland')
    parser.add_argument('--anomaly_source_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_repeat', type=int, default=1)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument('--perlin_max_scale', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--latent_res", type=int, default=64)
    # step 3. preparing accelerator')

    # step 4. model
    parser.add_argument("--beta_scale_factor", type=float, default=0.4)
    # lr
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--do_dist_loss", action='store_true')
    parser.add_argument("--dist_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_cls_train", action='store_true')
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)
    parser.add_argument("--anormal_weight", type=float, default=1.0)
    parser.add_argument('--normal_weight', type=float, default=1.0)
    import ast


    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--do_object_detect", action='store_true')
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    # step 7. inference check
    parser.add_argument("--scheduler_linear_start", type=float, default=0.00085)
    parser.add_argument("--scheduler_linear_end", type=float, default=0.012)
    parser.add_argument("--sample_sampler", type=str, default="ddim", choices=["ddim", "pndm", "lms", "euler",
                                                                               "euler_a", "heun", "dpm_2", "dpm_2_a",
                                                                               "dpmsolver", "dpmsolver++",
                                                                               "dpmsingle", "k_lms", "k_euler",
                                                                               "k_euler_a", "k_dpm_2", "k_dpm_2_a", ], )
    parser.add_argument("--scheduler_schedule", type=str, default="scaled_linear",
                        choices=["scaled_linear", "linear", "cosine", "cosine_warmup", ], )
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--unet_inchannels", type=int, default=9)
    parser.add_argument("--back_token_separating", action='store_true')
    parser.add_argument("--min_timestep", type=int, default=0)
    parser.add_argument("--max_timestep", type=int, default=1000)
    parser.add_argument("--down_dim", type=int)
    parser.add_argument("--noise_type", type=str)
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, worst quality, bad anatomy, bad composition, poor, low effort")
    parser.add_argument("--anomal_src_more", action='store_true')
    parser.add_argument("--without_background", action='store_true')
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--use_pe_pooling", action='store_true')
    parser.add_argument("--d_dim", default=320, type=int)
    parser.add_argument("--do_concat", action='store_true')

    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    main(args)