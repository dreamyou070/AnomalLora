import os
import argparse, torch
from model.lora import LoRANetwork
from attention_store import AttentionStore
from model.unet import unet_passing_argument
from utils.original_attention_control import register_attention_control, add_attn_argument, passing_argument
from accelerate import Accelerator
from model.tokenizer import load_tokenizer
from utils import prepare_dtype
from utils.model_utils import get_input_ids
from PIL import Image
from model.lora import LoRAInfModule
from utils.image_utils import load_image, image2latent
import numpy as np
from model.diffusion_model import load_target_model
from model.pe import PositionalEmbedding
from safetensors.torch import load_file
from torch import nn
def main(args):

    print(f'\n step 1. accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with,
                              project_dir='log')

    print(f'\n step 2. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype,
                                                   accelerator)

    if args.use_position_embedder:
        position_embedder = PositionalEmbedding(max_len=args.latent_res * args.latent_res,
                                                d_model=args.d_dim)

    print(f'\n step 2. accelerator and device')
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 3. inference')
    models = os.listdir(args.network_folder)
    network = LoRANetwork(text_encoder=text_encoder,
                          unet=unet,
                          lora_dim=args.network_dim,
                          alpha=args.network_alpha,
                          module_class=LoRAInfModule)
    network.apply_to(text_encoder, unet, True, True)
    raw_state_dict = network.state_dict()
    raw_state_dict_orig = raw_state_dict.copy()

    for model in models:

        network_model_dir = os.path.join(args.network_folder, model)
        lora_name, ext = os.path.splitext(model)
        lora_epoch = int(lora_name.split('-')[-1])

        parent = os.path.split(args.network_folder)[0]
        pe_base_dir = os.path.join(parent, f'position_embedder')

        pretrained_pe_dir = os.path.join(pe_base_dir, f'position_embedder_{lora_epoch}.safetensors')
        position_embedder_state_dict = load_file(pretrained_pe_dir)
        position_embedder.load_state_dict(position_embedder_state_dict)
        position_embedder.to(accelerator.device, dtype=weight_dtype)

        # [1] recon base folder
        parent, _ = os.path.split(args.network_folder)
        recon_base_folder = os.path.join(parent, 'reconstruction')
        os.makedirs(recon_base_folder, exist_ok=True)
        print(f'recon_base_folder: {recon_base_folder}')

        lora_base_folder = os.path.join(recon_base_folder, f'lora_epoch_{lora_epoch}')
        os.makedirs(lora_base_folder, exist_ok=True)

        for thred in args.threds :
            thred_folder = os.path.join(lora_base_folder, f'thred_{thred}')
            os.makedirs(thred_folder, exist_ok=True)

            check_base_folder = os.path.join(thred_folder, f'my_check')
            os.makedirs(check_base_folder, exist_ok=True)
            answer_base_folder = os.path.join(thred_folder, f'scoring/{args.obj_name}/test')
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

                    if accelerator.is_main_process:

                        with torch.no_grad():
                            img = load_image(rgb_img_dir, 512, 512)
                            vae_latent = image2latent(img, vae, weight_dtype)
                            input_ids, attention_mask = get_input_ids(tokenizer, args.prompt)
                            controller = AttentionStore()
                            register_attention_control(unet, controller)
                            for k in anomal_detecting_state_dict.keys():
                                raw_state_dict[k] = anomal_detecting_state_dict[k]
                            network.load_state_dict(raw_state_dict)
                            network.to(accelerator.device, dtype=weight_dtype)
                            encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device))["last_hidden_state"]
                            model_kwargs = {"position_embedder": position_embedder}
                            unet(vae_latent, 0, encoder_hidden_states,
                                 trg_layer_list=args.trg_layer_list, noise_type=position_embedder,
                                 **model_kwargs)
                            query_dict, key_dict = controller.query_dict, controller.key_dict
                            controller.reset()
                            query_list, key_list = [], []
                            for trg_layer in args.trg_layer_list:
                                query = query_dict[trg_layer][0]  # head, pix_num, dim
                                res = int(query.shape[1] ** 0.5)
                                query = query.view(-1, res, res, query.shape[-1])
                                query = query.permute(0, 3, 1, 2)  # 8, 320, 64, 64
                                query = nn.functional.interpolate(query, size=(64, 64), mode="bilinear")
                                query_list.append(query)
                                key = key_dict[trg_layer][0]  # head, pix_num, dim
                                key_list.append(key)
                            query = torch.cat(query_list, dim=1).permute(0, 2, 3, 1)
                            query = query.view(-1, 64 * 64, query.shape[-1])  # head, 64*64, 3520    # "dim": [12288],
                            key = torch.cat(key_list, dim=2)  # head, 77,    3520
                            attention_probs = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1],
                                                                        dtype=query.dtype, device=query.device), query,
                                                            key.transpose(-1, -2), beta=0).softmax(dim=-1)
                            cls_score, trigger_score = attention_probs[:, :, 0].squeeze(), attention_probs[:, :,
                                                                                           1].squeeze()  # pix_num
                            pix_num = cls_score.shape[0]
                            res = int(pix_num ** 0.5)
                            cls_map = cls_score.unsqueeze(0).view(res, res)
                            cls_map_pil = Image.fromarray((255*cls_map).cpu().detach().numpy().astype(np.uint8)).resize((org_h, org_w))
                            cls_map_pil.save(os.path.join(save_base_folder, f'{name}_cls_map.png'))

                            trigger_map = trigger_score.unsqueeze(0).view(res, res)
                            trigger_map_pil = Image.fromarray((255*trigger_map).cpu().detach().numpy().astype(np.uint8)).resize((org_h, org_w))
                            trigger_map_pil.save(os.path.join(save_base_folder, f'{name}_trigger_map.png'))

                            gt_img_save_dir = os.path.join(save_base_folder, f'{name}_gt.png')
                            Image.open(gt_img_dir).resize((org_h, org_w)).save(gt_img_save_dir)
                            for k in raw_state_dict_orig.keys():
                                raw_state_dict[k] = raw_state_dict_orig[k]
                            network.load_state_dict(raw_state_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int, default=64)
    parser.add_argument('--network_alpha', type=float, default=4)
    parser.add_argument('--network_folder', type=str)
    parser.add_argument("--lowram", action="store_true", )
    # step 4. dataset and dataloader
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument('--data_path', type=str,
                        default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bagel')
    # step 6
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--prompt", type=str, default="bagel", )
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--latent_res", type=int, default=64)
    parser.add_argument("--truncating", action='store_true')
    # step 8. test
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--threds", type=arg_as_list,default=[0.5])
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--use_pe_pooling", action='store_true')
    parser.add_argument("--d_dim", default=320, type=int)
    parser.add_argument("--thred", default=0.5, type=float)
    parser.add_argument("--image_classification_layer", type=str)
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--do_local_self_attn", action='store_true')
    parser.add_argument("--only_local_self_attn", action='store_true')
    parser.add_argument("--window_size", type=int, default=7)
    add_attn_argument(parser)
    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    main(args)