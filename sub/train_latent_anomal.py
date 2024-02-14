import os
import argparse, torch
from model.diffusion_model import load_SD_model
from model.tokenizer import load_tokenizer
from model.lora import LoRANetwork
from model.segmentation_model import SegmentationSubNetwork, DimentionChanger
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
from sub.attention_control import register_attention_control,add_attn_argument, passing_argument
from utils import get_epoch_ckpt_name, save_model
import time
import json
from torch import nn


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


def mahal(u, v, cov):
    delta = u - v
    cov_inv = cov.T
    m = torch.dot(delta, torch.matmul(cov_inv, delta))
    return torch.sqrt(m)

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
    print(f' (2.1) stable diffusion model')
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet = load_SD_model(args)
    vae_scale_factor = 0.18215
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                    num_train_timesteps=1000, clip_sample=False)
    print(f' (2.2) LoRA network')
    network = LoRANetwork(text_encoder=text_encoder, unet=unet, lora_dim = args.network_dim, alpha = args.network_alpha)
    print(f' (2.3) segmentation model')
    out_dim = 3
    dim_changer = DimentionChanger(in_dim=320, out_dim = out_dim)
    seg_model = SegmentationSubNetwork(in_channels=out_dim * 2, out_channels=1, )

    print(f' (2.2) attn controller')
    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f'\n step 3. optimizer')
    print(f' (3.1) lora optimizer')
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    print(f' (3.2) seg optimizer')
    segmentation_trainable_params = [{"params" : seg_model.parameters(), "lr" : args.seg_lr},
                                     {"params" : dim_changer.parameters(), "lr" : args.seg_lr}]
    optimizer_seg = torch.optim.AdamW(segmentation_trainable_params)

    loss_focal = BinaryFocalLoss()
    loss_smL1 = nn.SmoothL1Loss()

    print(f'\n step 4. dataset and dataloader')
    obj_dir = os.path.join(args.data_path, args.obj_name)
    train_dir = os.path.join(obj_dir, "train")
    root_dir = os.path.join(train_dir, "good/rgb")
    args.anomaly_source_path = os.path.join(args.data_path, "anomal_source")
    dataset = MVTecDRAEMTrainDataset(root_dir=root_dir,
                                     anomaly_source_path=args.anomaly_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer,
                                     caption=args.trigger_word,
                                     use_perlin=True,
                                     num_repeat = args.num_repeat,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f'\n step 5. lr')
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType.COSINE_WITH_RESTARTS]
    num_training_steps = len(dataloader) * args.num_epochs
    num_cycles = args.lr_scheduler_num_cycles
    lr_scheduler = schedule_func(optimizer, num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=num_training_steps, num_cycles=num_cycles, )
    seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg,
                                            T_max=10, eta_min=0, last_epoch=- 1, verbose=False)

    print(f'\n step 6. accelerator and device')
    weight_dtype, save_dtype = prepare_dtype(args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with, project_dir=args.logging_dir,)
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
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler, = accelerator.prepare(unet,
                                                            text_encoder, network, optimizer, dataloader, lr_scheduler)
        # unet, text_encoder, network, optimizer, optimizer_seg, train_dataloader, lr_scheduler, seg_scheduler,loss_focal, loss_smL1 = accelerator.prepare(
        # unet, text_encoder, network, optimizer, optimizer_seg, dataloader, lr_scheduler,seg_scheduler,loss_focal, loss_smL1)
    elif args.train_unet:
        unet, network, optimizer, optimizer_seg, train_dataloader, lr_scheduler, seg_scheduler,loss_focal, loss_smL1 = accelerator.prepare(unet, network, optimizer,
                                          optimizer_seg, dataloader, lr_scheduler, seg_scheduler,loss_focal, loss_smL1)
        text_encoder.to(accelerator.device,dtype=weight_dtype)
    elif args.train_text_encoder:
        text_encoder, network, optimizer, optimizer_seg, train_dataloader, lr_scheduler,seg_scheduler,loss_focal, loss_smL1 = accelerator.prepare(text_encoder, network,
                                optimizer, optimizer_seg,dataloader, lr_scheduler, seg_scheduler,loss_focal, loss_smL1)
        unet.to(accelerator.device,dtype=weight_dtype)

    print(f'\n step 7. Train!')
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []

    scheduler_cls = get_scheduler(args.sample_sampler, False)[0]
    scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps,
                              beta_start=args.scheduler_linear_start,
                              beta_end=args.scheduler_linear_end,
                              beta_schedule=args.scheduler_schedule)


    m = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.num_epochs}")

        for step, batch in enumerate(train_dataloader):
            loss = 0
            with torch.no_grad():
                latents = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample() # 1, 4, 64, 64
                latents = latents * vae_scale_factor # [1,4,64,64]
            with torch.set_grad_enabled(True) :
                input_ids = batch["input_ids"].to(accelerator.device) # batch, 77 sen len
                enc_out = text_encoder(input_ids)       # batch, 77, 768
                encoder_hidden_states = enc_out["last_hidden_state"]
            noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
            with accelerator.autocast():
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states,
                                  trg_layer_list=args.trg_layer_list, noise_type='perlin').sample
            ############################################# 1. task loss #################################################
            if args.do_task_loss:
                target = noise
                task_loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                task_loss = task_loss.mean([1, 2, 3]).mean()
                task_loss = task_loss * args.task_loss_weight

            ############################################ 2. Dist Loss ##################################################
            query_dict = controller.query_dict
            attn_dict = controller.step_store
            map_dict = controller.map_dict
            controller.reset()
            normal_feat_list = []
            anormal_feat_list = []

            dist_loss, normal_dist_loss, anomal_dist_loss = 0, 0, 0
            attn_loss, normal_loss, anomal_loss = 0, 0, 0
            segmentation_loss = 0
            loss_dict = {}

            for trg_layer in args.trg_layer_list:
                # (1) query dist
                normal_query, anomal_query = query_dict[trg_layer][0].chunk(2, dim=0)
                mu, cov, idx               = query_dict[trg_layer][1]
                anomal_map = map_dict[trg_layer][0].squeeze(0)

                anomal_map_vector = anomal_map.flatten()
                normal_query, anomal_query = normal_query.squeeze(0), anomal_query.squeeze(0) # pix_num, dim
                pix_num = normal_query.shape[0]
                for pix_idx in range(pix_num):
                    anormal_feat = anomal_query[pix_idx].squeeze(0)
                    normal_feat = normal_query[pix_idx].squeeze(0)
                    anomal_flag = anomal_map_vector[pix_idx]
                    if anomal_flag == 1:
                        anormal_feat_list.append(anormal_feat.unsqueeze(0))
                    else :
                        if args.more_generalize:
                            normal_feat_list.append(anormal_feat.unsqueeze(0))
                    normal_feat_list.append(normal_feat.unsqueeze(0))
                n_features = torch.cat(normal_feat_list, dim=0)
                n_features = torch.index_select(n_features, 1, idx)
                n_dists = [mahal(feat, mu, cov) for feat in n_features]
                normal_dist_mean = torch.tensor(n_dists).mean()
                if len(anormal_feat_list) > 0:
                    a_features = torch.cat(anormal_feat_list, dim=0)
                    a_features = torch.index_select(a_features, 1, idx)
                    a_dists = [mahal(feat, mu, cov) for feat in a_features]
                    anormal_dist_mean = torch.tensor(a_dists).mean()
                else :
                    anormal_dist_mean = torch.zeros_like(normal_dist_mean).to(normal_dist_mean.device)

                total_dist = normal_dist_mean + anormal_dist_mean
                normal_dist_loss = (normal_dist_mean / total_dist) ** 2
                normal_dist_loss = normal_dist_loss * args.dist_loss_weight
                dist_loss += normal_dist_loss.requires_grad_()

                anormal_dist_loss = (1 - (anormal_dist_mean / total_dist)) ** 2
                anormal_dist_loss = anormal_dist_loss * args.dist_loss_weight
                dist_loss += anormal_dist_loss.requires_grad_()

                ################## ---------------------- ################## ---------------------- ##################

                attention_score = attn_dict[trg_layer][0]  # 2, pix_num, 2

                cls_score, trigger_score = attention_score.chunk(2, dim=-1)
                normal_cls_score, anormal_cls_score = cls_score.chunk(2, dim=0)  #
                normal_trigger_score, anormal_trigger_score = trigger_score.chunk(2, dim=0)

                normal_cls_score, anormal_cls_score = normal_cls_score.squeeze(), anormal_cls_score.squeeze()  # head, pix_num
                normal_trigger_score, anormal_trigger_score = normal_trigger_score.squeeze(), anormal_trigger_score.squeeze()

                anomal_map_vector = anomal_map_vector.unsqueeze(0).repeat(normal_cls_score.shape[0], 1).to(anormal_cls_score.device)
                #if args.more_generalize:
                normal_map_vector = 1 - anomal_map_vector
                anormal_cls_score, anormal_trigger_score = anormal_cls_score * anomal_map_vector, anormal_trigger_score * anomal_map_vector
                normal_cls_score_, normal_trigger_score_ = normal_cls_score * normal_map_vector, normal_trigger_score * normal_map_vector

                anormal_cls_score, anormal_trigger_score = anormal_cls_score.mean(dim=0), anormal_trigger_score.mean(dim=0)
                normal_cls_score_, normal_trigger_score_ = normal_cls_score_.mean(dim=0), normal_trigger_score_.mean(dim=0)
                normal_cls_score, normal_trigger_score = normal_cls_score.mean(dim=0), normal_trigger_score.mean(dim=0)
                total_score = torch.ones_like(normal_cls_score)

                normal_cls_score_loss = (normal_cls_score / total_score) ** 2
                normal_trigger_score_loss = (1 - (normal_trigger_score / total_score)) ** 2
                anormal_cls_score_loss = (1 - (anormal_cls_score / total_score)) ** 2
                anormal_trigger_score_loss = (anormal_trigger_score / total_score) ** 2
                normal_cls_score_loss_ = (normal_cls_score_ / total_score) ** 2
                normal_trigger_score_loss_ = (1-(normal_trigger_score_ / total_score)) ** 2

                attn_loss += args.normal_weight * normal_trigger_score_loss + args.anormal_weight * anormal_trigger_score_loss
                if args.do_anomal_sample_normal_loss :
                    attn_loss +=  args.normal_weight * normal_trigger_score_loss_

                normal_loss += normal_trigger_score_loss # + normal_trigger_score_loss_
                anomal_loss += anormal_trigger_score_loss

                if args.do_cls_train :
                    attn_loss += args.normal_weight * normal_cls_score_loss + args.anormal_weight * anormal_cls_score_loss
                    if args.do_anomal_sample_normal_loss :
                        attn_loss += args.normal_weight * normal_cls_score_loss_
                    normal_loss += normal_cls_score_loss
                    if args.do_anomal_sample_normal_loss :
                        normal_loss += normal_cls_score_loss_
                    anomal_loss += anormal_cls_score_loss

                ############################################ 3. segmentation net ###########################################
                """
                normal_query = dim_changer(normal_query)  # [batch, pix_nujm, dum -> batch, pix_num, 3]
                normal_query = einops.rearrange(normal_query, 'b (h w) c -> b c h w', h=64, w=64)

                anormal_query = dim_changer(anomal_query) # batch, pix_num, 3
                anormal_query = einops.rearrange(anormal_query, 'b (h w) c -> b c h w', h=64, w=64)

                seg_input = torch.cat((normal_query, anormal_query), dim=1)  # [batch, 6, 64,64]
                pred_mask = seg_model(seg_input)  # [batch, 1, 64, 64]
                pred_mask_trg = anomal_map.unsqueeze(0).unsqueeze(0)

                focal_loss = loss_focal(pred_mask, pred_mask_trg)
                smL1_loss = loss_smL1(pred_mask, pred_mask_trg)
                segmentation_loss += focal_loss + 5 * smL1_loss
                """
            ############################################ 4. total Loss ##################################################
            if args.do_task_loss:
                loss += task_loss
                loss_dict['task_loss'] = task_loss.item()
            if args.do_dist_loss:
                loss += dist_loss
                #loss_dict['dist_loss'] = dist_loss.item()
                loss_dict['normal_dist_loss'] = normal_dist_loss.item()
                loss_dict['anormal_dist_loss'] = anormal_dist_loss.item()
            if args.do_attn_loss:
                loss += attn_loss.mean()
                #loss_dict['attn_loss'] = attn_loss.mean().item()
                loss_dict['normal_loss'] = normal_loss.mean().item()
                loss_dict['anomal_loss'] = anomal_loss.mean().item()
            #loss_dict['segmentation_loss'] = segmentation_loss.mean().item()
            #loss += segmentation_loss.mean()

            current_loss = loss.detach().item()
            if epoch == args.start_epoch :
                loss_list.append(current_loss)
            else:
                epoch_loss_total -= loss_list[step]
                loss_list[step] = current_loss
            epoch_loss_total += current_loss
            avr_loss = epoch_loss_total / len(loss_list)
            loss_dict['avr_loss'] = avr_loss

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ### 4.1 logging
            #accelerator.log(loss_dict, step=global_step)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                controller.reset()
            if is_main_process:
                progress_bar.set_postfix(**loss_dict)
            if global_step >= args.max_train_steps:
                break

        # ----------------------------------------------- Epoch Final ----------------------------------------------- #
        accelerator.wait_for_everyone()
        ### 4.2 sampling
        if is_main_process :
            ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
            save_model(args, ckpt_name, accelerator.unwrap_model(network), save_dtype)

            pipeline = AnomalyDetectionStableDiffusionPipeline(vae=vae, text_encoder=text_encoder,
                                                               tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                                                               safety_checker=None, feature_extractor=None,
                                                               requires_safety_checker=False,
                                                               random_vector_generator=None, trg_layer_list=None)
            latents = pipeline(prompt=args.trigger_word,
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
            #controller.reset()
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
    parser.add_argument("--task_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_dist_loss", action='store_true')
    parser.add_argument("--dist_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_cls_train", action='store_true')
    parser.add_argument('--normal_weight', type=float, default=1.0)
    parser.add_argument('--anormal_weight', type=float, default=1.0)
    parser.add_argument("--do_anomal_sample_normal_loss", action='store_true')
    parser.add_argument("--trg_layer_list", type=arg_as_list, )
    parser.add_argument("--save_model_as",type=str,default="safetensors",
                        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],)
    parser.add_argument("--output_name", type=str, default=None,
                        help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名")
    parser.add_argument("--general_training", action='store_true')
    parser.add_argument("--trigger_word", type = str, default = "good")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--more_generalize", action='store_true')
    add_attn_argument(parser)
    args = parser.parse_args()
    passing_argument(args)
    main(args)