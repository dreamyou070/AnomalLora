# !/bin/bash

port_number=59845
obj_name='bagel'
caption='good'

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../gradient_based_image_generation.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_folder "../../result/${obj_name}/latent_anomal/latent_caption_bagel_down_dim_320_more_generalize/models" \
 --data_path "../../../MyData/anomaly_detection/MVTec3D-AD/${obj_name}/test" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']" \
 --down_dim 160 \
 --prompt "${caption}"