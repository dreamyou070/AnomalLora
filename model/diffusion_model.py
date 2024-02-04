import os
from diffusers import StableDiffusionPipeline
from model.unet import UNet2DConditionModel
def load_SD_model(args):

    name_or_path = args.pretrained_model_name_or_path
    pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
    text_encoder, vae, unet = pipe.text_encoder, pipe.vae, pipe.unet
    del pipe
    original_unet = UNet2DConditionModel(unet.config.sample_size,
                                         unet.config.attention_head_dim,
                                         unet.config.cross_attention_dim,
                                         unet.config.use_linear_projection,
                                         unet.config.upcast_attention,)
    original_unet.load_state_dict(unet.state_dict())
    unet = original_unet
    return text_encoder, vae, unet