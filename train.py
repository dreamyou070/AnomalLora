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

def main(args) :

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

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
    args.anomaly_source_path = os.path.join(train_dir, "anomal")
    dataset = MVTecDRAEMTrainDataset(root_dir=root_dir,
                                     anomaly_source_path=args.anomaly_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer, )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

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
        accelerator.print(f"\nepoch {epoch + 1}/{args.num_epochs}")
        for step, batch in enumerate(train_dataloader):

            with torch.no_grad():
                latents = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample()
                anomal_latents = vae.encode(batch['augmented_image'].to(dtype=weight_dtype)).latent_dist.sample()
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                    anomal_latents = torch.where(torch.isnan(anomal_latents),
                                                 torch.zeros_like(anomal_latents),anomal_latents)
                latents = latents * vae_scale_factor
                anomal_latents = anomal_latents * vae_scale_factor
                input_latents = torch.cat([latents, anomal_latents], dim=0)
            with torch.set_grad_enabled(True) :
                input_ids = batch["input_ids"].to(accelerator.device) # batch, 77 sen len
                encoder_hidden_states = text_encoder(input_ids)       # batch, 77, 768
                print(f'encoder_hidden_states.shape: {encoder_hidden_states}')
            input_text_encoder_conds = torch.cat([encoder_hidden_states,encoder_hidden_states], dim=0)
            noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(args, noise_scheduler,input_latents)
            with accelerator.autocast():
                noise_pred = unet(noisy_latents, timesteps, input_text_encoder_conds,
                                                           trg_indexs_list=args.trg_layer_list, mask_imgs=None).sample
                normal_noise_pred, anomal_noise_pred = torch.chunk(noise_pred, 2, dim=0)

            ############################################################################################################
            if args.do_task_loss:
                target = noise.chunk(2, dim=0)[0]
                loss = torch.nn.functional.mse_loss(normal_noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])
                task_loss = loss.mean()
                task_loss = task_loss * args.task_loss_weight

            ############################################ 2. Dist Loss ##################################################
            query_dict = controller.query_dict
            attn_dict = controller.step_store
            controller.reset()
            normal_feats, anormal_feats = [], []
            dist_loss, attn_loss = 0, 0
            anomal_mask = batch['anomaly_mask'].flatten().squeeze(0)
            for trg_layer in args.trg_layer_list:
                normal_query, anomal_query = query_dict[trg_layer].chunk(2, dim=0)
                normal_query, anomal_query = normal_query.squeeze(0), anomal_query.squeeze(0) # pix_num, dim
                pix_num = normal_query.shape[0]

                for pix_idx in range(pix_num):
                    normal_feat = normal_query[pix_idx].unsqueeze(0)
                    anomal_feat = anomal_query[pix_idx].unsqueeze(0)
                    anomal_flag = anomal_mask[pix_idx]
                    if anomal_flag == 0:
                        anormal_feats.append(anomal_feat.unsqueeze(0))
                    normal_feats.append(normal_feat.unsqueeze(0))
                normal_feats = torch.cat(normal_feats, dim=0)
                anormal_feats = torch.cat(anormal_feats, dim=0)
                normal_mu = torch.mean(normal_feats, dim=0)
                normal_cov = torch.cov(normal_feats.transpose(0, 1))

                def mahal(u, v, cov):
                    delta = u - v
                    m = torch.dot(delta, torch.matmul(cov, delta))
                    return torch.sqrt(m)

                normal_mahalanobis_dists = [mahal(feat, normal_mu, normal_cov) for feat in normal_feats]
                anormal_mahalanobis_dists = [mahal(feat, normal_mu, normal_cov) for feat in anormal_feats]
                normal_dist_mean = torch.tensor(normal_mahalanobis_dists).mean()
                anormal_dist_mean = torch.tensor(anormal_mahalanobis_dists).mean()
                total_dist = normal_dist_mean + anormal_dist_mean
                normal_dist_loss = (normal_dist_mean / total_dist) ** 2
                anormal_dist_loss = (1 - (anormal_dist_mean / total_dist)) ** 2
                dist_loss += normal_dist_loss.requires_grad_() + anormal_dist_loss.requires_grad_()



            # [1] img
            """
            img = batch['image']
            
            
            idx = batch['idx']
            input_ids = batch['input_ids']

            print(f'img.shape: {img.shape}')
            print(f'anomal_mask.shape: {anomal_mask.shape}')
            print(f'synthetic_img.shape: {synthetic_img.shape}')
            print(f'idx.shape: {idx}')
            print(f'input_ids.shape: {input_ids}')
            """


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
    parser.add_argument("--trg_layer_list", type=arg_as_list, )
    args = parser.parse_args()
    main(args)