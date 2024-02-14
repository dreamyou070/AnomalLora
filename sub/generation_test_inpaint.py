import os
import argparse, torch
from model.diffusion_model import load_SD_model
from model.tokenizer import load_tokenizer
from model.lora import LoRANetwork
from accelerate import Accelerator
from utils import prepare_dtype
from utils.scheduling_utils import get_scheduler


def main(args) :

    print(f'\n step 1. model')
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet = load_SD_model(args)

    print(f'\n step 2. accelerator and device')
    weight_dtype, save_dtype = prepare_dtype(args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                             mixed_precision=args.mixed_precision,log_with=args.log_with, project_dir='log')
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device,dtype=weight_dtype)
    scheduler_cls = get_scheduler(args.sample_sampler, False)[0]
    scheduler = scheduler_cls(num_train_timesteps=args.scheduler_timesteps, beta_start=args.scheduler_linear_start,
                              beta_end=args.scheduler_linear_end, beta_schedule=args.scheduler_schedule)

    print(f'\n step 3. object_detector network')
    from safetensors.torch import load_file

    print(f'\n step 4. inference')
    models = os.listdir(args.network_folder)

    from model.lora import LoRAInfModule
    network = LoRANetwork(text_encoder=text_encoder, unet=unet, lora_dim=args.network_dim, alpha=args.network_alpha,
                          module_class=LoRAInfModule)
    network.apply_to(text_encoder, unet, True, True)
    raw_state_dict = network.state_dict()
    raw_state_dict_orig = raw_state_dict.copy()

    for model in models:
        network_model_dir = os.path.join(args.network_folder, model)
        lora_name, ext = os.path.splitext(model)
        lora_epoch = int(lora_name.split('-')[-1])

        # [1] recon base folder
        parent, _ = os.path.split(args.network_folder)
        recon_base_folder = os.path.join(parent, 'generation')

        anomal_detecting_state_dict = load_file(network_model_dir)

        if accelerator.is_main_process:
            with torch.no_grad():
                for k in raw_state_dict_orig.keys():
                    raw_state_dict[k] = raw_state_dict_orig[k]
                network.load_state_dict(raw_state_dict)
                for k in anomal_detecting_state_dict.keys():
                    raw_state_dict[k] = anomal_detecting_state_dict[k]
                network.load_state_dict(raw_state_dict)
                network.to(accelerator.device, dtype=weight_dtype)
                # -------------------------------------------------- #
                from diffusers import StableDiffusionInpaintPipeline

                pipeline = StableDiffusionInpaintPipeline(vae=vae,text_encoder=text_encoder,tokenizer=tokenizer,
                                                          unet=unet,scheduler=scheduler,safety_checker=None,
                                                          feature_extractor=None, requires_safety_checker=False)
                from PIL import Image
                test_rgb_dir = os.path.join(args.data_path, f'{args.obj_name}/test/combined/rgb/000.png')
                test_gt_dir = os.path.join(args.data_path, f'{args.obj_name}/test/combined/gt/000.png')
                test_rgb_pil = Image.open(test_rgb_dir)
                test_gt_pil = Image.open(test_gt_dir)
                latent = pipeline(prompt='bagel',
                                   image=test_rgb_pil,
                                   mask_image=test_gt_pil,
                                   height=512, width=512, num_inference_steps=args.num_ddim_steps,
                                   guidance_scale=args.guidance_scale,
                                   negative_prompt=args.negative_prompt, )[-1]
                gen_image = pipeline.latents_to_image(latent)[0].resize((512,512))
                gen_image.save(os.path.join(recon_base_folder, f'{lora_name}.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int,default=64)
    parser.add_argument('--network_alpha', type=float,default=4)
    parser.add_argument('--network_folder', type=str)
    parser.add_argument('--object_detector_weight', type=str)

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
    add_attn_argument(parser)
    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    main(args)