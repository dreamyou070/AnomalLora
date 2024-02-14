import argparse, random, json
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
import torch
import os
from data.mvtec_sy import MVTecDRAEMTrainDataset
from model.diffusion_model import load_target_model, transform_models_if_DDP
from model.tokenizer import load_tokenizer
from utils import prepare_dtype
from utils.accelerator_utils import prepare_accelerator
from utils.model_utils import prepare_scheduler_for_custom_training
from utils.pipeline import AnomalyDetectionStableDiffusionPipeline
from utils.scheduling_utils import get_scheduler

vae_scale_factor = 0.18215


def call_unet(args, accelerator, unet, noisy_latents, timesteps,
              text_conds, batch, weight_dtype, trg_indexs_list, mask_imgs):
    noise_pred = unet(noisy_latents,
                      timesteps,
                      text_conds,
                      trg_layer_list=args.trg_layer_list,
                      noise_type=args.noise_type).sample
    return noise_pred


def main(args):
    print(f'\n step 1. setting')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(args.logging_dir, exist_ok=True)
    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    if args.seed is None: args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f'\n step 2. dataset')
    accelerator = prepare_accelerator(args)
    tokenizer = load_tokenizer(args)
    obj_dir = os.path.join(args.data_path, args.obj_name)

    test_dir = os.path.join(obj_dir, "test")
    folders = os.listdir(test_dir)
    for folder in folders:
        root_dir = os.path.join(test_dir, f"{folder}/rgb")
        args.anomaly_source_path = os.path.join(args.data_path, "anomal_source")
        dataset = MVTecDRAEMTrainDataset(root_dir=root_dir,
                                         anomaly_source_path=args.anomaly_source_path,
                                         resize_shape=[512, 512],
                                         tokenizer=tokenizer,
                                         caption=args.trigger_word,
                                         use_perlin=True,
                                         num_repeat=args.num_repeat,
                                         anomal_only_on_object=args.anomal_only_on_object,
                                         anomal_training=True)

        weight_dtype, save_dtype = prepare_dtype(args)
        vae_dtype = weight_dtype
        print(f' (4.1) base model')
        text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        unet.requires_grad_(False)
        unet.to(dtype=weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)
        unet, text_encoder, train_dataloader = accelerator.prepare(unet, text_encoder, train_dataloader)

        text_encoders = transform_models_if_DDP(text_encoders)
        unet = transform_models_if_DDP([unet])
        unet = unet[0]

        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=1000, clip_sample=False)
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

        scheduler_cls = get_scheduler(args.sample_sampler, False)[0]
        scheduler = scheduler_cls(num_train_timesteps=1000, beta_start=args.scheduler_linear_start,
                                  beta_end=args.scheduler_linear_end, beta_schedule=args.scheduler_schedule)
        pipeline = AnomalyDetectionStableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                                                           unet=unet, scheduler=scheduler, safety_checker=None,
                                                           feature_extractor=None,
                                                           requires_safety_checker=False, random_vector_generator=None,
                                                           trg_layer_list=None)

        img_save_base_dir = os.path.join(args.output_dir, 'noising_test')
        os.makedirs(img_save_base_dir, exist_ok=True)

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                name = batch['image_name'][0]
                # [1] original image
                latents = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample() # 1, 4, 64, 64
                latents = latents * vae_scale_factor  # [1,4,64,64]

                noise = torch.randn_like(latents, device=latents.device)

                random_times = [20,50, 100, 150, 300, 500]

                for random_time in random_times:

                    random_timestep = torch.tensor([random_time])
                    random_timestep = random_timestep.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, random_timestep)

                    # [2] augmented image
                    anomal_latents = vae.encode(batch['augmented_image'].to(dtype=weight_dtype)).latent_dist.sample()
                    anomal_latents = anomal_latents * vae_scale_factor
                    noisy_anomal_latents = noise_scheduler.add_noise(anomal_latents, noise, random_timestep)

                    origin_pil = pipeline.latents_to_image(latents)[0].resize((512, 512))
                    origin_noise_pil = pipeline.latents_to_image(noisy_latents)[0].resize((512, 512))
                    anomal_pil = pipeline.latents_to_image(anomal_latents)[0].resize((512, 512))
                    anomal_noise_pil = pipeline.latents_to_image(noisy_anomal_latents)[0].resize((512, 512))


                    origin_pil.save(os.path.join(img_save_base_dir, f'{name}_origin.png'))
                    origin_noise_pil.save(os.path.join(img_save_base_dir, f'{name}_origin_noise_{random_time}_timestep.png'))
                    anomal_pil.save(os.path.join(img_save_base_dir, f'{name}_anomal.png'))
                    anomal_noise_pil.save(os.path.join(img_save_base_dir, f'{name}_anomal_noise_{random_time}_timestep.png'))




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
    parser.add_argument('--trigger_word,', type=str)
    # step 3. preparing accelerator')
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    # step 4. model
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='facebook/diffusion-dalle')
    parser.add_argument("--network_weights", type=str, default=None,
                        help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None,
                        help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=64,
                        help="network dimensions (depends on each network) ")
    parser.add_argument("--network_alpha", type=float, default=1,
                        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)", )
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)", )
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value)")
    parser.add_argument("--lowram", action="store_true", )
    parser.add_argument("--sample_every_n_steps", type=int, default=None,
                        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する")
    parser.add_argument("--sample_every_n_epochs", type=int, default=None,
                        help="generate sample images every N epochs (overwrites n_steps)", )
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
    parser.add_argument("--scheduler_linear_start", type=float, default=0.00085)
    parser.add_argument("--scheduler_linear_end", type=float, default=0.012)
    # step 7. inference check
    parser.add_argument("--sample_sampler", type=str, default="ddim",
                        choices=["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver",
                                 "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2",
                                 "k_dpm_2_a", ], )
    parser.add_argument("--scheduler_schedule", type=str, default="scaled_linear",
                        choices=["scaled_linear", "linear", "cosine", "cosine_warmup", ], )
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--output_name", type=str, default=None, help="base name of trained model file ")
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors)", )
    parser.add_argument("--network_train_unet_only", action="store_true",
                        help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                        help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--dim_from_weights", action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim ", )
    parser.add_argument("--scale_weight_norms", type=float, default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ", )
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training ", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
    parser.add_argument("--only_object_position", action="store_true", )
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--resume_lora_training", action="store_true", )
    parser.add_argument("--back_training", action="store_true", )
    parser.add_argument("--back_weight", type=float, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--valid_data_dir", type=str)
    parser.add_argument("--task_loss_weight", type=float, default=0.5)
    parser.add_argument("--truncate_pad", action='store_true')
    parser.add_argument("--truncate_length", type=int, default=3)
    parser.add_argument("--anormal_sample_normal_loss", action='store_true')
    parser.add_argument("--do_task_loss", action='store_true')
    parser.add_argument("--do_dist_loss", action='store_true')
    parser.add_argument("--dist_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_cls_train", action='store_true')
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)
    parser.add_argument("--anormal_weight", type=float, default=1.0)
    parser.add_argument('--normal_weight', type=float, default=1.0)
    parser.add_argument("--clip_skip", type=int, default=None,
                        help="use output of nth layer from back of text encoder (n>=1)")
    parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                        help="max token length of text encoder (default for 75, 150 or 225) / text encoder", )
    parser.add_argument("--normal_with_back", action='store_true')
    parser.add_argument("--normal_dist_loss_squere", action='store_true')
    parser.add_argument("--background_with_normal", action='store_true')
    parser.add_argument("--background_weight", type=float, default=1)
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--add_random_query", action="store_true", )
    parser.add_argument("--unet_frozen", action="store_true", )
    parser.add_argument("--text_frozen", action="store_true", )
    parser.add_argument("--trigger_word", type=str, default='teddy bear, wearing like a super hero')
    parser.add_argument("--full_fp16", action="store_true",
                        help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients")
    # step 8. training
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument('--mahalanobis_loss_weight', type=float, default=1.0)
    parser.add_argument("--cls_training", action="store_true", )
    parser.add_argument("--background_loss", action="store_true")
    parser.add_argument("--average_mask", action="store_true", )
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, worst quality, bad anatomy, bad composition, poor, low effort")
    # extra
    parser.add_argument("--unet_inchannels", type=int, default=9)
    parser.add_argument("--back_token_separating", action='store_true')
    parser.add_argument("--more_generalize", action='store_true')
    parser.add_argument("--down_dim", type=int)
    parser.add_argument("--noise_type", type=str)
    parser.add_argument("--only_zero_timestep", action="store_true")
    parser.add_argument("--truncating", action="store_true")
    args = parser.parse_args()
    from model.unet import unet_passing_argument
    from sub.attention_control import passing_argument
    unet_passing_argument(args)
    passing_argument(args)
    args = parser.parse_args()
    main(args)