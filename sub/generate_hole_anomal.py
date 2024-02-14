import argparse, random
from accelerate.utils import set_seed
from PIL import Image
import torch
import os
from model.tokenizer import load_tokenizer
from model.unet import unet_passing_argument
from sub.attention_control import passing_argument
import numpy as np
import cv2
import skimage
from data.perlin import rand_perlin_2d_np

perlin_max_scale = 8
kernel_size = 5
def make_random_mask(height, width) -> np.ndarray:

    perlin_scale = perlin_max_scale
    min_perlin_scale = 4
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    noise = rand_perlin_2d_np(shape = (height, width),
                              res = (perlin_scalex, perlin_scaley))

    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_pil = Image.fromarray(mask).convert('L')
    mask_np = np.array(mask_pil) / 255  # height, width, [0,1]

    return mask_np, mask_pil

def main(args):

    print(f'\n step 1. setting')
    if args.seed is None: args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f'\n step 2. dataset')
    tokenizer = load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
    obj_dir = os.path.join(args.data_path, args.obj_name)
    train_dir = os.path.join(obj_dir, "train")

    good_data_dir = os.path.join(train_dir, "good")


    bad_data_dir = os.path.join(train_dir, "bad")
    os.makedirs(bad_data_dir, exist_ok=True)
    bad_data_rgb_dir = os.path.join(bad_data_dir, "rgb")
    bad_data_gt_dir = os.path.join(bad_data_dir, "gt")
    bad_data_object_mask_dir = os.path.join(bad_data_dir, "object_mask")
    os.makedirs(bad_data_rgb_dir, exist_ok=True)
    os.makedirs(bad_data_gt_dir, exist_ok=True)
    os.makedirs(bad_data_object_mask_dir, exist_ok=True)

    good_rgb_dir = os.path.join(good_data_dir, "rgb")
    good_object_mask_dir = os.path.join(good_data_dir, "object_mask")
    good_gt_dir = os.path.join(good_data_dir, "gt")
    good_background_dir = os.path.join(good_data_dir, "background")
    os.makedirs(good_rgb_dir, exist_ok=True)

    good_images = os.listdir(good_rgb_dir)

    h, w = 512, 512

    for image in good_images:

        good_img_dir = os.path.join(good_rgb_dir, image)
        good_img_pil = Image.open(good_img_dir).resize((h,w))
        good_img_np = np.array(good_img_pil)

        back_dir = os.path.join(good_background_dir, image)
        back_pil = Image.open(back_dir).resize((h,w))
        back_np = np.array(back_pil)

        # [1] mask mask
        dtype = good_img_np.dtype
        object_mask_dir = os.path.join(good_object_mask_dir, image)
        object_mask_pil = Image.open(object_mask_dir).convert('L').resize((h,w))
        object_mask_np = np.array(object_mask_pil)
        object_mask_np = np.where(object_mask_np == 0, 0, 1)  # 1 = object, 0 = background
        while True:
            anomal_mask_np, anomal_mask_pil = make_random_mask(h,w)
            anomal_mask_np = np.where(anomal_mask_np == 0, 0, 1)  # strict anomal (0, 1
            anomal_mask_np = anomal_mask_np * object_mask_np
            if anomal_mask_np.sum() > 0:
                break
        final_mask_np = np.repeat(np.expand_dims(anomal_mask_np, axis=2), 3, axis=2).astype(dtype)  # 1 = anomal, 0 = normal

        pseudo_anomal_img = ((1 - final_mask_np) * good_img_np + (final_mask_np) * back_np ).astype(np.uint8)

        pseudo_anomal_img_pil = Image.fromarray(pseudo_anomal_img).resize((w,h))
        pseudo_anomal_mask_pil = Image.fromarray((anomal_mask_np * 255).astype(np.uint8)).convert('L').resize((w,h))

        pseudo_anomal_img_pil.save(os.path.join(bad_data_dir, "rgb", image))
        pseudo_anomal_mask_pil.save(os.path.join(bad_data_dir, "gt", image))
        pseudo_anomal_object_mask_pil = Image.fromarray((object_mask_np).astype(np.uint8)).convert('L').resize((w, h))
        pseudo_anomal_object_mask_pil.save(os.path.join(bad_data_dir, "object_mask", image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--wandb_api_key', type=str, default='output')
    parser.add_argument('--wandb_project_name', type=str, default='bagel')
    # step 2. dataset
    parser.add_argument('--data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument('--anomaly_source_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_repeat', type=int, default=1)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument('--perlin_max_scale', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--latent_res", type=int, default=64)
    # step 3. preparing accelerator')
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--full_fp16", action="store_true",
                        help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients")
    # step 4. model
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='facebook/diffusion-dalle')
    parser.add_argument("--clip_skip", type=int, default=None,
                        help="use output of nth layer from back of text encoder (n>=1)")
    parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                        help="max token length of text encoder (default for 75, 150 or 225) / text encoder", )
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_dim", type=int, default=64,help="network dimensions (depends on each network) ")
    parser.add_argument("--network_alpha", type=float, default=4, help="alpha for LoRA weight scaling, default 1 ", )
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step", )
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value)")
    parser.add_argument("--dim_from_weights", action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim ", )
    parser.add_argument("--scale_weight_norms", type=float, default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ", )
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training ", )
    # step 5. optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                help="AdamW , AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov, "
              "SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, "
                             "DAdaptLion, DAdaptSGD, AdaFactor", )
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="use 8bit AdamW optimizer (requires bitsandbytes)", )
    parser.add_argument("--use_lion_optimizer", action="store_true",
                        help="use Lion optimizer (requires lion-pytorch)", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping")
    parser.add_argument("--optimizer_args", type=str, default=None, nargs="*",
                        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") ', )
    # lr
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module")
    parser.add_argument("--lr_scheduler_args", type=str, default=None, nargs="*",
                        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）', )
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", help="scheduler to use for lr")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler (default is 0)", )
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                        help="Number of restarts for cosine scheduler with restarts / cosine with restarts", )
    parser.add_argument("--lr_scheduler_power", type=float, default=1,
                        help="Polynomial power for polynomial scheduler / polynomial", )
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--unet_lr', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--train_unet', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    # training
    parser.add_argument("--total_normal_thred", type=float, default=0.5)
    parser.add_argument("--lowram", action="store_true", )
    parser.add_argument("--sample_every_n_steps", type=int, default=None,
                        help="generate sample images every N steps ")
    parser.add_argument("--sample_every_n_epochs", type=int, default=None,
                        help="generate sample images every N epochs (overwrites n_steps)", )
    parser.add_argument("--output_name", type=str, default=None, help="base name of trained model file ")
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors)", )
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
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
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    # step 7. inference check
    parser.add_argument("--scheduler_linear_start", type=float, default=0.00085)
    parser.add_argument("--scheduler_linear_end", type=float, default=0.012)
    parser.add_argument("--sample_sampler", type=str, default="ddim",  choices=["ddim", "pndm", "lms", "euler",
                                             "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++",
                                        "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", ], )
    parser.add_argument("--scheduler_schedule", type=str, default="scaled_linear",
                                               choices=["scaled_linear", "linear", "cosine", "cosine_warmup", ], )
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--unet_inchannels", type=int, default=9)
    parser.add_argument("--back_token_separating", action='store_true')
    parser.add_argument("--min_timestep", type=int, default=0)
    parser.add_argument("--max_timestep", type=int, default=1000)
    parser.add_argument("--down_dim", type=int)
    parser.add_argument("--noise_type", type=str)
    parser.add_argument("--truncating", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--negative_prompt", type=str,
                             default="low quality, worst quality, bad anatomy, bad composition, poor, low effort")
    parser.add_argument("--anomal_src_more", action = 'store_true')
    parser.add_argument("--without_background", action='store_true')
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    main(args)