import os
import argparse, torch
from model.diffusion_model import load_SD_model, transform_models_if_DDP
from model.tokenizer import load_tokenizer
from model.lora import LoRANetwork
from model.segmentation_model import SegmentationSubNetwork
from data.mvtec_sy import MVTecDRAEMTrainDataset
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
from torch import nn
import einops
import torch.nn.functional as F
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def main(args) :

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    print(f' ---------- output dir = {output_dir} ----------')
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f'\n step 2. model')
    print(f' (2.1) stable diffusion model')
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet = load_SD_model(args)
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    vae_scale_factor = 0.18215
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                    num_train_timesteps=1000, clip_sample=False)
    print(f' (2.2) LoRA network')
    network = LoRANetwork(text_encoder=text_encoder, unet=unet, lora_dim = args.network_dim, alpha = args.network_alpha)
    network.apply_to(text_encoder, unet, args.train_unet, args.train_text_encoder)

    print(f'\n step 3. optimizer')
    print(f' (3.1) lora optimizer')
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    print(f'\n step 4. dataset and dataloader')
    obj_dir = os.path.join(args.data_path, args.obj_name)
    train_dir = os.path.join(obj_dir, "train")
    root_dir = os.path.join(train_dir, "good/rgb")
    args.anomaly_source_path = os.path.join(args.data_path, "anomal_source")
    dataset = MVTecDRAEMTrainDataset(root_dir=root_dir,
                                     anomaly_source_path=args.anomaly_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer,
                                     caption = args.trigger_word,
                                     use_perlin = True,
                                     num_repeat = args.num_repeat,
                                     anomal_only_on_object = args.anomal_only_on_object)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f'\n step 5. lr')
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType.COSINE_WITH_RESTARTS]
    num_training_steps = len(dataloader) * args.num_epochs
    lr_scheduler = schedule_func(optimizer, num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=num_training_steps, num_cycles=args.lr_scheduler_num_cycles)

    print(f'\n step 6. accelerator and device')
    weight_dtype, save_dtype = prepare_dtype(args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                    mixed_precision=args.mixed_precision, log_with=args.log_with, project_dir=args.logging_dir,)
    is_main_process = accelerator.is_main_process
    if args.full_fp16:
        assert (args.mixed_precision == "fp16"), "full_fp16 requires mixed precision='fp16'"
        accelerator.print("enable full fp16 training.")
        network.to(weight_dtype)
    elif args.full_bf16:
        assert (args.mixed_precision == "bf16"), "full_bf16 requires mixed precision='bf16' / mixed_precision='bf16'"
        accelerator.print("enable full bf16 training.")
        network.to(weight_dtype)

    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)

    print(f'\n step 7. training preparing')
    if args.train_unet and args.train_text_encoder:
        unet, text_encoder, network, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, network, optimizer, dataloader, lr_scheduler)
        text_encoders = [text_encoder]
    elif args.train_unet:
        unet, network, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, network, optimizer, dataloader, lr_scheduler)
        text_encoder.to(accelerator.device)
    elif args.train_text_encoder:
        text_encoder, network, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, network, optimizer, dataloader, lr_scheduler)
        text_encoders = [text_encoder]
        unet.to(accelerator.device, dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
    else:
        network, optimizer, dataloader, lr_scheduler = accelerator.prepare(network, optimizer, dataloader, lr_scheduler)
    text_encoders = transform_models_if_DDP(text_encoders)
    unet, network = transform_models_if_DDP([unet, network])
    if args.gradient_checkpointing:
        unet.train()
        for text_encoder in text_encoders:
            text_encoder.train()
            if args.train_text_encoder:
                text_encoder.text_model.embeddings.requires_grad_(True)
        if not args.train_text_encoder:  # train U-Net only
            unet.parameters().__next__().requires_grad_(True)
    else:
        unet.eval()
        for text_encoder in text_encoders:
            text_encoder.eval()
    network.prepare_grad_etc(text_encoder, unet)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 7. Train!')
    train_steps = args.num_epochs * len(dataloader)
    progress_bar = tqdm(range(train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    for epoch in range(args.start_epoch, args.num_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.num_epochs}")

        for step, batch in enumerate(dataloader):
            loss = torch.tensor(0.0, dtype=weight_dtype, device=accelerator.device)
            # --------------------------------------------- Task Loss --------------------------------------------- #
            with torch.set_grad_enabled(True):
                input_ids = batch["input_ids"].to(accelerator.device)  # batch, 77 sen len
                enc_out = text_encoder(input_ids)  # batch, 77, 768
                encoder_hidden_states = enc_out["last_hidden_state"]

            if args.do_task_loss:
                with torch.no_grad():
                    latents = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample() # 1, 4, 64, 64
                    latents = latents * vae_scale_factor  # [1,4,64,64]

                noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(args, noise_scheduler,latents)
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states,
                                      trg_layer_list=args.trg_layer_list, noise_type=None).sample
                target = noise
                task_loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                task_loss = task_loss.mean([1, 2, 3])
                task_loss = task_loss.mean()
                task_loss = task_loss * args.task_loss_weight




            loss += task_loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step >= args.max_train_steps:
                break
        # ----------------------------------------------- Epoch Final ----------------------------------------------- #
        accelerator.wait_for_everyone()
        ### 4.2 sampling
        if is_main_process :
            ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
            save_model(args, ckpt_name, accelerator.unwrap_model(network), save_dtype)
            scheduler_cls = get_scheduler(args.sample_sampler, False)[0]
            scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps,
                                      beta_start=args.scheduler_linear_start, beta_end=args.scheduler_linear_end,
                                      beta_schedule=args.scheduler_schedule)
            pipeline = AnomalyDetectionStableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                                        unet=unet, scheduler=scheduler,safety_checker=None, feature_extractor=None,
                                        requires_safety_checker=False, random_vector_generator=None, trg_layer_list=None)
            latents = pipeline(prompt=batch['caption'],
                               height=512, width=512,  num_inference_steps=args.num_ddim_steps,
                               guidance_scale=args.guidance_scale, negative_prompt=args.negative_prompt, )
            gen_img = pipeline.latents_to_image(latents[-1])[0].resize((512, 512))
            img_save_base_dir = args.output_dir + "/sample"
            os.makedirs(img_save_base_dir, exist_ok=True)
            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            num_suffix = f"e{epoch:06d}"
            img_filename = (f"{ts_str}_{num_suffix}_seed_{args.seed}.png")
            gen_img.save(os.path.join(img_save_base_dir,img_filename))
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--wandb_api_key', type=str,default='output')
    parser.add_argument('--wandb_project_name', type=str,default='bagel')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int,default=64)
    parser.add_argument('--network_alpha', type=float,default=4)
    parser.add_argument('--network_weights', type=str)
    # 3. optimizer
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--unet_lr', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--seg_lr', type=float, default=1e-5)
    # step 4. dataset and dataloader
    parser.add_argument('--data_path', type=str,
                        default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument('--anomaly_source_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_repeat', type=int, default=1)
    # step 5. lr
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--lr_scheduler_num_cycles', type=int, default=1)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    # step 6
    parser.add_argument('--train_unet', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"],)
    parser.add_argument("--save_precision",  type=str,  default=None,
                         choices=[None, "float", "fp16", "bf16"],)
    parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients")
    parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients")
    parser.add_argument("--gradient_checkpointing", action="store_true",help="enable gradient checkpointing")
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
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--change_dist_loss", action='store_true')
    parser.add_argument("--normal_dist_loss_squere", action='store_true')

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
    parser.add_argument("--trigger_word", type = str, default = "good")
    parser.add_argument("--unet_inchannels", type=int, default=4)
    args = parser.parse_args()
    from model.unet import unet_passing_argument
    unet_passing_argument(args)
    args = parser.parse_args()
    main(args)