import os
import argparse, torch
from model.diffusion_model import load_SD_model
from model.tokenizer import load_tokenizer
from model.lora import LoRANetwork
from data.mvtec import MVTecDRAEMTrainDataset
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from accelerate import Accelerator
from utils import prepare_dtype
from utils.pipeline import AnomalyDetectionStableDiffusionPipeline
from utils.scheduling_utils import get_scheduler

def main(args) :

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    args.log_dir = os.path.join(output_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f'\n step 2. model')
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet = load_SD_model(args)
    network = LoRANetwork(text_encoder=text_encoder, unet=unet,
                          lora_dim = args.network_dim, alpha = args.network_alpha,)
    if args.network_weights is not None:
        network.load_weights(args.network_weights)

    print(f'\n step 3. optimizer')
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    print(f'\n step 4. dataset and dataloader')
    dataset = MVTecDRAEMTrainDataset(root_dir=args.data_path + args.obj_name + "/train/good/",
                                     #anomaly_source_path=args.anomaly_source_path,
                                anomaly_source_path=args.data_path, resize_shape=[512,512],tokenizer=tokenizer,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=16)

    print(f'\n step 5. lr')
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType.COSINE_WITH_RESTARTS]
    num_training_steps = len(dataloader) * args.num_epochs
    num_cycles = args.lr_scheduler_num_cycles
    lr_scheduler = schedule_func(optimizer, num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=num_training_steps, num_cycles=num_cycles, )

    print(f'\n step 6. accelerator and device')
    weight_dtype, save_dtype = prepare_dtype(args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                            mixed_precision=args.mixed_precision, log_with=args.log_with,project_dir=args.logging_dir,)
    is_main_process = accelerator.is_main_process
    vae.requires_grad_(False)
    vae.to(dtype=weight_dtype)
    vae.to(accelerator.device)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
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
    network.prepare_grad_etc(text_encoder, unet)

    print(f'\n step 7. inference check')
    scheduler_cls = get_scheduler(args.sample_sampler, False)[0]
    scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps,
                              beta_start=args.scheduler_linear_start,
                              beta_end=args.scheduler_linear_end,
                              beta_schedule=args.scheduler_schedule)
    pipeline = AnomalyDetectionStableDiffusionPipeline(vae=vae,
                                                       text_encoder=text_encoder,
                                                       tokenizer=tokenizer, unet=unet,
                                                       scheduler=scheduler,
                                                       safety_checker=None,
                                                       feature_extractor=None,
                                                       requires_safety_checker=False, )
    # input_ids = batch["input_ids"].to(accelerator.device)
    # encoder_hidden_states = get_hidden_states(args, input_ids, tokenizer, text_encoders, weight_dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
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
                        default=r'../../../MyData/anomaly_detection/MVTec/')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument('--anomaly_source_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    # step 5. lr
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr_scheduler_num_cycles', type=int, default=1)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    # step 6
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--save_precision",type=str,default=None,choices=[None, "float", "fp16", "bf16"],)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--log_with",type=str,default=None,choices=["tensorboard", "wandb", "all"],)
    # step 7. inference check
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--sample_sampler",type=str,default="ddim",
                        choices=["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver",
                                 "dpmsolver++","dpmsingle","k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a",],)
    parser.add_argument("--scheduler_timesteps",type=int,default=1000,)
    parser.add_argument("--scheduler_linear_start",type=float,default=0.00085)
    parser.add_argument("--scheduler_linear_end",type=float,default=0.012,)
    parser.add_argument("--scheduler_schedule",type=str,default="scaled_linear",
                        choices=["scaled_linear","linear","cosine","cosine_warmup",],)
    args = parser.parse_args()
    main(args)
