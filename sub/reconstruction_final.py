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

    print(f'\n step 3. object_detector network')
    from safetensors.torch import load_file

    print(f'\n step 4. inference')
    models = os.listdir(args.network_folder)
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
        recon_base_folder = os.path.join(parent, 'reconstruction')
        os.makedirs(recon_base_folder, exist_ok=True)

        lora_base_folder = os.path.join(recon_base_folder, f'lora_epoch_{lora_epoch}')
        os.makedirs(lora_base_folder, exist_ok=True)

        check_base_folder = os.path.join(lora_base_folder, f'my_check')
        os.makedirs(check_base_folder, exist_ok=True)
        answer_base_folder = os.path.join(lora_base_folder, f'scoring/{args.obj_name}/test')
        os.makedirs(answer_base_folder, exist_ok=True)

        anomal_detecting_state_dict = load_file(network_model_dir)

        test_img_folder = args.data_path
        anomal_folders = os.listdir(test_img_folder)
        for anomal_folder in anomal_folders:

            answer_anomal_folder = os.path.join(answer_base_folder, anomal_folder)
            os.makedirs(answer_anomal_folder, exist_ok=True)

            save_base_folder = os.path.join(check_base_folder, anomal_folder)
            os.makedirs(save_base_folder, exist_ok=True)
            anomal_folder_dir = os.path.join(test_img_folder, anomal_folder)
            rgb_folder = os.path.join(anomal_folder_dir, 'rgb')
            gt_folder = os.path.join(anomal_folder_dir, 'gt')
            rgb_imgs = os.listdir(rgb_folder)
            for rgb_img in rgb_imgs:
                name, ext = os.path.splitext(rgb_img)
                rgb_img_dir = os.path.join(rgb_folder, rgb_img)
                org_h, org_w = Image.open(rgb_img_dir).size


                img_dir = os.path.join(save_base_folder, f'{name}_org{ext}')
                Image.open(rgb_img_dir).resize((org_h, org_w)).save(img_dir)


                gt_img_dir = os.path.join(gt_folder, f'{name}.png')

                # --------------------------------- gen cross attn map ---------------------------------------------- #
                if accelerator.is_main_process:
                    with torch.no_grad():
                        img = load_image(rgb_img_dir, 512, 512)
                        vae_latent = image2latent(img, vae, weight_dtype)
                        input_ids, attention_mask = get_input_ids(tokenizer, args.prompt)

                        controller = AttentionStore()
                        register_attention_control(unet, controller)

                        # [1] anomal detection  --------------------------------------------------------------------- #
                        for k in anomal_detecting_state_dict.keys():
                            raw_state_dict[k] = anomal_detecting_state_dict[k]
                        network.load_state_dict(raw_state_dict)
                        network.to(accelerator.device, dtype=weight_dtype)
                        encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device))["last_hidden_state"]
                        unet(vae_latent, 0, encoder_hidden_states,
                             trg_layer_list=args.trg_layer_list,)
                        attn_dict = controller.step_store
                        query_dict = controller.query_dict
                        controller.reset()
                        thred = len(args.trg_layer_list) / 2

                        map_list = []
                        for layer_name in args.trg_layer_list:
                            attn_map = attn_dict[layer_name][0]
                            if attn_map.shape[0] != 8:
                                attn_map = attn_map.chunk(2, dim=0)[0]

                            if args.truncating :
                                cls_map, trigger_map = attn_map.chunk(2, dim=-1)  # head, pix_num
                            else :
                                trigger_map = attn_map[:,:,1].squeeze()
                            trigger_map = (trigger_map.squeeze()).mean(dim=0)  #
                            map_list.append(trigger_map)

                            pix_num = trigger_map.shape[0]
                            res = int(pix_num ** 0.5)

                            normal_map = torch.where(trigger_map > 0.5, 1, trigger_map).squeeze()

                            normal_map = normal_map.unsqueeze(0)
                            normal_map = normal_map.view(res, res)
                            normal_map_pil = Image.fromarray(
                                normal_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))
                            normal_map_pil.save(os.path.join(save_base_folder, f'{name}_normal_score_map_{layer_name}.png'))

                            anomaly_map = (1 - normal_map) * 255
                            anomal_np = anomaly_map.cpu().detach().numpy().astype(np.uint8)
                            anomaly_map_pil = Image.fromarray(anomal_np).resize((org_h, org_w))
                            anomaly_map_pil.save(os.path.join(save_base_folder, f'{name}_anomaly_score_map_{layer_name}.png'))

                            #binary_map = torch.where(trigger_map > 0.5, 1, 0).squeeze()
                            #binary_map = binary_map.unsqueeze(0)
                            #binary_map = binary_map.view(res, res)
                            #binary_pil = Image.fromarray(
                            #    binary_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))
                            #binary_pil.save(os.path.join(save_base_folder, f'{name}_attn_map_{layer_name}.png'))

                            anomaly_map_pil.save(os.path.join(answer_anomal_folder, f'{name}.tiff'))

                        #map = torch.stack(map_list, dim=0)
                        #map = map.mean(dim=0) # pix_num
                        #normal_score_map = torch.where(map > 0.5, 1, map)
                        #anomal_score_map = 1 - normal_score_map
                        #anomaly_score_pil = Image.fromarray(anomal_score_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))
                        #anomaly_mask_save_dir = os.path.join(save_base_folder, f'{name}{ext}')
                        #anomaly_score_pil.save(anomaly_mask_save_dir)


                        #answer_dir = os.path.join(answer_anomal_folder, f'{name}.tiff')
                        #anomaly_score_pil.save(answer_dir)

                        """
                        binary_map = torch.where(map > thred, 1, 0).squeeze()
                        pix_num = binary_map.shape[0]
                        res = int(pix_num ** 0.5)
                        binary_map = binary_map.unsqueeze(0)
                        binary_map = binary_map.view(res, res)
                        binary_pil = Image.fromarray(binary_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))
                        binary_pil.save(os.path.join(save_base_folder, f'{name}_attn_map_concat.png'))

                        # [2] object detection --------------------------------------------------------------------- #
                        
                        for k in raw_state_dict_orig.keys():
                            raw_state_dict[k] = raw_state_dict_orig[k]
                        network.load_state_dict(raw_state_dict)
                        for k in object_detecting_state_dict.keys():
                            raw_state_dict[k] = object_detecting_state_dict[k]
                        network.load_state_dict(raw_state_dict)
                        encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device))["last_hidden_state"]
                        object_detect_trg_layer = ['up_blocks_3_attentions_2_transformer_blocks_0_attn2']
                        unet(vae_latent,0,encoder_hidden_states,trg_layer_list=object_detect_trg_layer)
                        attn_dict = controller.step_store
                        controller.reset()
                        attn_map = attn_dict[object_detect_trg_layer[0]][0]
                        if attn_map.shape[0] != 8:
                            attn_map = attn_map.chunk(2, dim=0)[0]
                        if args.back_token_separating:
                            cks_map, trigger_map, _ = attn_map.chunk(3, dim=-1)  # head, pix_num
                        else :
                            cks_map, trigger_map = attn_map.chunk(2, dim=-1)  # head, pix_num
                        trigger_map = (trigger_map.squeeze()).mean(dim=0) #
                        object_map = torch.where(trigger_map > 0.5, 1, 0).squeeze() # object = 1
                        pix_num = object_map.shape[0]
                        res = int(pix_num ** 0.5)
                        object_map = object_map.unsqueeze(0).view(res, res) # object = 1, background = 0
                        object_pil = Image.fromarray(object_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))
                        object_pil.save(os.path.join(save_base_folder, f'{name}_object_map_{layer_name}.png'))
                        

                        anormal_map = torch.where((object_map > 0) & (binary_map == 0), 1, 0) # object and anomal
                        recon_map = 1 - anormal_map
                        recon_pil = Image.fromarray(recon_map.cpu().detach().numpy().astype(np.uint8) * 255).resize(
                            (org_h, org_w))
                        recon_pil.save(os.path.join(save_base_folder, f'{name}_recon_map_{layer_name}.png'))

                        # [3] image generation --------------------------------------------------------------------- #
                        pipeline = AnomalyDetectionStableDiffusionPipeline(vae=vae, text_encoder=text_encoder,
                                    tokenizer=tokenizer,unet=unet,scheduler=scheduler,safety_checker=None,
                                    feature_extractor=None, requires_safety_checker=False, random_vector_generator=None,
                                                                           trg_layer_list=None)
                        latents = pipeline(prompt=args.prompt, height=512, width=512,
                                           num_inference_steps=args.num_ddim_steps,
                                           guidance_scale=args.guidance_scale,negative_prompt=args.negative_prompt,
                                           reference_image=vae_latent, mask=recon_map)
                        controller.reset()
                        recon_latent = latents[-1]
                        recon_image = pipeline.latents_to_image(recon_latent)[0].resize((org_h, org_w))
                        img_dir = os.path.join(save_base_folder, f'{name}_recon{ext}')
                        recon_image.save(img_dir)

                        # [4] anomal map ----------------------------------------------------------------------------- #
                        for k in raw_state_dict_orig.keys():
                            raw_state_dict[k] = raw_state_dict_orig[k]
                        network.load_state_dict(raw_state_dict)
                        for k in anomal_detecting_state_dict.keys():
                            raw_state_dict[k] = anomal_detecting_state_dict[k]
                        network.load_state_dict(raw_state_dict)

                        org_image = pipeline.latents_to_image(vae_latent)[0].resize((org_h, org_w))
                        img_dir = os.path.join(save_base_folder, f'{name}_org{ext}')
                        org_image.save(img_dir)
                        # -------------------------------------------------------------------------------------------- #
                        unet(vae_latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list)
                        org_query_dict = controller.query_dict
                        org_query = org_query_dict[args.trg_layer_list[0]][0].squeeze(0) # pix_num, dim
                        org_query = org_query / (torch.norm(org_query, dim=1, keepdim=True))
                        controller.reset()
                        # -------------------------------------------------------------------------------------------- #
                        # (3) recon : recon_latent
                        unet(recon_latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list)
                        recon_query_dict = controller.query_dict
                        recon_query = recon_query_dict[args.trg_layer_list[0]][0].squeeze(0) # pix_num, dim
                        recon_query = recon_query / (torch.norm(recon_query, dim=1, keepdim=True))

                        # (3) anomaly score
                        nomaly_score = (org_query @ recon_query.T).cpu() # pix_num, pix_num
                        anomaly_score = (1-torch.diag(nomaly_score))
                        anomaly_score = anomaly_score / anomaly_score.max()
                        anomaly_score = anomaly_score.unsqueeze(0) # [1, pix_num]
                        anomaly_score = anomaly_score.view(res, res)
                        anomaly_score = anomaly_score.numpy()
                        anomaly_score_pil = Image.fromarray((anomaly_score * 255).astype(np.uint8)).resize((org_h, org_w))
                        anomaly_mask_save_dir = os.path.join(save_base_folder, f'{name}{ext}')
                        anomaly_score_pil.save(anomaly_mask_save_dir)
                        """

                        # [5] save anomaly score --------------------------------------------------------------------- #
                        gt_img_save_dir = os.path.join(save_base_folder, f'{name}_gt.png')
                        Image.open(gt_img_dir).resize((org_h, org_w)).save(gt_img_save_dir)

                        #tiff_anomaly_mask_save_dir = os.path.join(evaluate_class_dir, f'{name}.tiff')
                        #anomaly_score_pil.save(tiff_anomaly_mask_save_dir)
                        #network.restore()
                        for k in raw_state_dict_orig.keys():
                            raw_state_dict[k] = raw_state_dict_orig[k]
                        network.load_state_dict(raw_state_dict)
                        #network.apply_to(text_encoder, unet, True, True)




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
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--use_pe_pooling", action='store_true')
    parser.add_argument("--d_dim", default=320, type=int)
    parser.add_argument("--do_concat", action='store_true')
    add_attn_argument(parser)
    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    main(args)