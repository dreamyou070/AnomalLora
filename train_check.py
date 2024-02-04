import os
import argparse, torch
from model.diffusion_model import load_SD_model
from model.tokenizer import load_tokenizer
from model.lora import LoRANetwork
from data.mvtec import MVTecDRAEMTrainDataset
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from diffusers import DDPMScheduler
from accelerate import Accelerator
from utils import prepare_dtype
from utils.model_utils import get_noise_noisy_latents_and_timesteps
from attention_store import AttentionStore
from utils.pipeline import AnomalyDetectionStableDiffusionPipeline
from utils.scheduling_utils import get_scheduler
from tqdm import tqdm
from utils.attention_control import register_attention_control
from utils import get_epoch_ckpt_name, save_model
import time
import json

def main(args) :

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f'\n step 2. model')
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet = load_SD_model(args)
    vae_scale_factor = 0.18215
    network = LoRANetwork(text_encoder=text_encoder, unet=unet, lora_dim = args.network_dim, alpha = args.network_alpha)
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                                                            num_train_timesteps=1000, clip_sample=False)
    print(f' (2.2) attn controller')
    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f'\n step 3. optimizer')
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    print(f'\n step 4. dataset and dataloader')
    obj_dir = os.path.join(args.data_path, args.obj_name)
    train_dir = os.path.join(obj_dir, "train")
    root_dir = os.path.join(train_dir, "good/rgb")
    if args.general_training :
        caption = 'good'
        args.anomaly_source_path = os.path.join(train_dir, "anomal_general")
    else :
        caption = args.obj_name
        args.anomaly_source_path = os.path.join(train_dir, "anomal")
    dataset = MVTecDRAEMTrainDataset(root_dir=root_dir,
                                     anomaly_source_path=args.anomaly_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer,
                                     caption = caption,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    print(f'\n step 5. lr')
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType.COSINE_WITH_RESTARTS]
    num_training_steps = len(dataloader) * args.num_epochs
    num_cycles = args.lr_scheduler_num_cycles
    lr_scheduler = schedule_func(optimizer, num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=num_training_steps, num_cycles=num_cycles, )

    print(f'\n step 6. accelerator and device')
    weight_dtype, save_dtype = prepare_dtype(args)

    if args.log_with in ["wandb", "all"]:
        try:
            import wandb
        except ImportError:
            raise ImportError("No wandb / wandb がインストールされていないようです")
        os.environ["WANDB_DIR"] = args.logging_dir
        if args.wandb_api_key is not None:
            wandb.login(key=args.wandb_api_key)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with,
                              project_dir=args.logging_dir,)
    is_main_process = accelerator.is_main_process
    vae.requires_grad_(False)
    vae.to(dtype=weight_dtype)
    vae.to(accelerator.device)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    print(f' (6.2) network with stable diffusion model')
    network.prepare_grad_etc(text_encoder, unet)
    network.apply_to(text_encoder, unet, True, True)
    if args.network_weights is not None:
        network.load_weights(args.network_weights)
    if args.train_unet and args.train_text_encoder:
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                                                 unet, text_encoder, network, optimizer, dataloader, lr_scheduler)
    elif args.train_unet:
        unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, network, optimizer,
                                                                                       dataloader, lr_scheduler)
        text_encoder.to(accelerator.device,dtype=weight_dtype)
    elif args.train_text_encoder:
        text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(text_encoder, network,
                                                                                 optimizer, dataloader, lr_scheduler)
        unet.to(accelerator.device,dtype=weight_dtype)

    print(f'\n step 7. Train!')

    for epoch in range(0, args.num_epochs):
        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.num_epochs}")
        ### 4.2 sampling
        if is_main_process :
            ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
            unwrapped_nw = accelerator.unwrap_model(network)
            save_model(args, ckpt_name, unwrapped_nw, save_dtype)
            scheduler_cls = get_scheduler(args.sample_sampler, False)[0]
            scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps,
                                      beta_start=args.scheduler_linear_start,
                                      beta_end=args.scheduler_linear_end,
                                      beta_schedule=args.scheduler_schedule)
            pipeline = AnomalyDetectionStableDiffusionPipeline(vae=vae,
                                                               text_encoder=text_encoder,
                                                               tokenizer=tokenizer,
                                                               unet=unet,
                                                               scheduler=scheduler,
                                                               safety_checker=None,
                                                               feature_extractor=None,
                                                               requires_safety_checker=False,
                                                               random_vector_generator=None,
                                                               trg_layer_list=None)
            anomal_concepts = ['crack', 'cut', 'hole', 'contaminate', 'crash', 'dirty', 'bad', 'torn', 'crumpled',
                               'deformed']
            for adjective in anomal_concepts:
                adjective_base_folder = os.path.join(args.anomal_base_dir, adjective)
                os.makedirs(adjective_base_folder, exist_ok=True)
                gen_caption = f'the concept of {adjective}'
                for i in range(1):
                    latents = pipeline(prompt=gen_caption,
                                       height=512, width=512,
                                       num_inference_steps=args.num_ddim_steps,
                                       guidance_scale=args.guidance_scale,
                                       negative_prompt=args.negative_prompt, )
                    gen_img = pipeline.latents_to_image(latents[-1])[0].resize((512, 512))
                    img_save_base_dir = args.output_dir + "/sample"
                    os.makedirs(img_save_base_dir, exist_ok=True)
                    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                    num_suffix = f"e{epoch:06d}"
                    img_filename = (f"{ts_str}_{num_suffix}_seed_{args.seed}.png")
                    gen_img.save(os.path.join(img_save_base_dir,img_filename))
                    controller.reset()
        accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--wandb_api_key', type=str,
                        default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int,default=64)
    parser.add_argument('--network_alpha', type=float,default=4)
    parser.add_argument('--network_weights', type=str)
    # 3. optimizer
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--unet_lr', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    # step 4. dataset and dataloader
    parser.add_argument('--data_path', type=str,
                        default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument('--anomaly_source_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    # step 5. lr
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr_scheduler_num_cycles', type=int, default=1)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    # step 6
    parser.add_argument('--train_unet', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--save_precision",type=str,default=None,choices=[None, "float", "fp16", "bf16"],)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,)
    parser.add_argument("--log_with",type=str,default=None,choices=["tensorboard", "wandb", "all"],)

    # step 7. inference check
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--sample_sampler",type=str,default="ddim",
                        choices=["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver",
                                 "dpmsolver++","dpmsingle","k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a",],)
    parser.add_argument("--scheduler_timesteps",type=int,default=1000,)
    parser.add_argument("--scheduler_linear_start",type=float,default=0.00085)
    parser.add_argument("--scheduler_linear_end",type=float,default=0.012,)
    parser.add_argument("--scheduler_schedule",type=str,default="scaled_linear",
                        choices=["scaled_linear","linear","cosine","cosine_warmup",],)
    parser.add_argument("--prompt", type=str, default="bagel",)
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, worst quality, bad anatomy, bad composition, poor, low effort")
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--do_task_loss", action='store_true')
    parser.add_argument("--task_loss_weight", type=float, default=0.5)
    parser.add_argument("--do_dist_loss", action='store_true')
    parser.add_argument("--dist_loss_weight", type=float, default = 1)
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_cls_train", action='store_true')
    parser.add_argument('--normal_weight', type=float, default=1.0)
    parser.add_argument('--anormal_weight', type=float, default=1.0)
    parser.add_argument("--trg_layer_list", type=arg_as_list, )
    parser.add_argument("--save_model_as",type=str,default="safetensors",
                        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],)
    parser.add_argument("--output_name", type=str, default=None,
                        help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名")
    parser.add_argument("--general_training", action='store_true')
    args = parser.parse_args()
    main(args)