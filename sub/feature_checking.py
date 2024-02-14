import os
import argparse, torch
from model.lora import LoRANetwork
from attention_store import AttentionStore
from sub.attention_control import register_attention_control
from accelerate import Accelerator
from model.tokenizer import load_tokenizer
from utils import prepare_dtype
from utils.scheduling_utils import get_scheduler
from utils.model_utils import get_input_ids
from PIL import Image
from model.lora import LoRAInfModule
from utils.image_utils import load_image, image2latent
import numpy as np
from model.diffusion_model import load_target_model


def main(args) :

    print(f'\n step 1. accelerator')
    weight_dtype, save_dtype = prepare_dtype(args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision, log_with=args.log_with, project_dir='log')

    print(f'\n step 2. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    tokenizer = load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
    vae_dtype = weight_dtype
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    print(f'\n step 2. accelerator and device')
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device,dtype=weight_dtype)
    scheduler_cls = get_scheduler(args.sample_sampler, False)[0]
    scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps, beta_start=args.scheduler_linear_start,
                              beta_end=args.scheduler_linear_end, beta_schedule=args.scheduler_schedule)

    print(f'\n step 4. inference')
    print(f' (1) loading network')
    network = LoRANetwork(text_encoder=text_encoder, unet=unet, lora_dim=args.network_dim, alpha=args.network_alpha,
                          module_class=LoRAInfModule)
    network.apply_to(text_encoder, unet, True, True)
    from safetensors.torch import load_file
    pretrained_lora_dir = r'/home/dreamyou070/AnomalLora/result/bagel/train_query_optimizing/models/epoch-000031.safetensors'
    anomal_detecting_state_dict = load_file(pretrained_lora_dir)
    network.load_state_dict(anomal_detecting_state_dict, strict=False)
    network.to(accelerator.device, dtype=weight_dtype)

    print(f' (2) start with image')
    img_base_dir = r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/bagel/test/crack'
    org_img_dir = os.path.join(img_base_dir, 'rgb', '005.png')
    gt_img_dir = os.path.join(img_base_dir, 'gt', '005.png')

    with torch.no_grad():
        print(f' [1] img')
        img = load_image(org_img_dir, 512, 512)
        vae_latent = image2latent(img, vae, weight_dtype)
        print(f' [2] text')
        input_ids, attention_mask = get_input_ids(tokenizer, 'bagel')
        encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device))["last_hidden_state"]
        print(f' [3] unet')
        controller = AttentionStore()
        register_attention_control(unet, controller)
        print(f'args.trg_layer_list: {args.trg_layer_list}')
        unet(vae_latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, )

        # get query
        attn_dict = controller.step_store
        query_dict = controller.query_dict
        controller.reset()

        query = query_dict[args.trg_layer_list[0]][0].squeeze().to('cpu') # pix_num, dim
        gt_img = Image.open(gt_img_dir).convert('L').resize((64,64))
        gt_img_np = np.array(gt_img) # 64,64
        anomal_position = torch.tensor(np.where(gt_img_np > 0, 1, 0)).flatten() # 4096
        normal_position = 1 - anomal_position

        pix_num, dim = query.shape
        anomal_features, normal_features = [], []
        for i in range(pix_num):
            anomal_flag = anomal_position[i]
            if anomal_flag == 1: # anomal
                anomal_feature = query[i].unsqueeze(0)
                anomal_features.append(anomal_feature)
            else : # normal
                normal_feature = query[i].unsqueeze(0)
                normal_features.append(normal_feature)
        anomal_features = torch.cat(anomal_features, dim=0) # anomal_num, dim
        normal_features = torch.cat(normal_features, dim=0) # normal_num, dim

        normal_mu = torch.mean(normal_features, dim=0)
        normal_cov = torch.cov(normal_features.transpose(0, 1))

        def mahal(u, v, cov):
            delta = u - v
            m = torch.dot(delta, torch.matmul(cov, delta))
            return torch.sqrt(m)

        normal_mahalanobis_dists = [mahal(feat, normal_mu, normal_cov) for feat in normal_features]
        anomal_mahalanobis_dists = [mahal(feat, normal_mu, normal_cov) for feat in anomal_features]

        # [5] plot histogram
        import matplotlib.pyplot as plt
        plt.hist(normal_mahalanobis_dists, bins=100, alpha=0.5, label='normal')
        plt.savefig(os.path.join(args.output_dir, f'histogram/normal.png'))
        plt.cla()
        plt.hist(anomal_mahalanobis_dists, bins=100, alpha=0.5, label='anomal')
        plt.savefig(os.path.join(args.output_dir, f'histogram/anomal.png'))
        plt.cla()

        print(f' [4] down dim')
        from random import sample
        d = 100
        idx = torch.tensor(sample(range(0, 320), d))
        normal_features_down_dim = torch.index_select(normal_features, 1, idx)
        anomal_features_down_dim = torch.index_select(anomal_features, 1, idx)
        normal_mu_down_dim = torch.mean(normal_features_down_dim, dim=0)
        normal_cov_down_dim = torch.cov(normal_features_down_dim.transpose(0, 1))
        normal_mahalanobis_dists_down_dim = [mahal(feat, normal_mu_down_dim, normal_cov_down_dim) for feat in normal_features_down_dim]
        anomal_mahalanobis_dists_down_dim = [mahal(feat, normal_mu_down_dim, normal_cov_down_dim) for feat in anomal_features_down_dim]
        plt.hist(normal_mahalanobis_dists_down_dim, bins=100, alpha=0.5, label='normal')
        plt.savefig(os.path.join(args.output_dir, f'histogram/normal_down_dim_{d}.png'))
        plt.cla()
        plt.hist(anomal_mahalanobis_dists_down_dim, bins=100, alpha=0.5, label='anomal')
        plt.savefig(os.path.join(args.output_dir, f'histogram/anomal_down_dim_{d}.png'))
        plt.cla()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int,default=64)
    parser.add_argument('--network_alpha', type=float,default=4)
    parser.add_argument('--network_folder', type=str)
    parser.add_argument('--object_detector_weight', type=str)
    parser.add_argument("--lowram", action="store_true", )
    # step 4. dataset and dataloader
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument('--data_path', type=str,
                        default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bagel')
    parser.add_argument('--anomaly_source_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    # step 5. lr
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr_scheduler_num_cycles', type=int, default=1)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    # step 6
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
    parser.add_argument("--truncating", action='store_true')
    # step 8. test
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--trg_layer_list", type=arg_as_list)
    parser.add_argument("--more_generalize", action='store_true')
    from sub.attention_control import add_attn_argument, passing_argument
    from model.unet import unet_passing_argument
    parser.add_argument("--unet_inchannels", type=int, default=4)
    parser.add_argument("--back_token_separating", action = 'store_true')
    add_attn_argument(parser)
    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    main(args)