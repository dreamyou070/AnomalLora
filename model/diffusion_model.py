import os
from diffusers import StableDiffusionPipeline, AutoencoderKL
from model.unet import UNet2DConditionModel
from model.diffusion_model_conversion import (load_checkpoint_with_text_encoder_conversion,
                    convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint)
from model.diffusion_model_config import (create_unet_diffusers_config,create_vae_diffusers_config)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig, logging

def load_SD_model(args):

    # [0]
    ckpt_path = args.pretrained_model_name_or_path
    device = args.device
    _, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path, device)

    # [1] unet
    unet_config = create_unet_diffusers_config()
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)
    unet = UNet2DConditionModel(**unet_config).to(device)
    info = unet.load_state_dict(converted_unet_checkpoint)

    # [2] vae
    vae_config = create_vae_diffusers_config()
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
    vae = AutoencoderKL(**vae_config).to(device)
    info = vae.load_state_dict(converted_vae_checkpoint)

    # [3] text_encoder
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint(state_dict)
    cfg = CLIPTextConfig(vocab_size=49408,hidden_size=768,intermediate_size=3072,num_hidden_layers=12,
                         num_attention_heads=12,max_position_embeddings=77,
                         hidden_act="quick_gelu",layer_norm_eps=1e-05,dropout=0.0,
                         attention_dropout=0.0,initializer_range=0.02,initializer_factor=1.0,
                         pad_token_id=1,bos_token_id=0,eos_token_id=2,
                         model_type="clip_text_model",projection_dim=768,torch_dtype="float32",)
    text_model = CLIPTextModel._from_config(cfg)
    converted_text_encoder_checkpoint.pop('text_model.embeddings.position_ids')
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)

    return text_model, vae, unet
