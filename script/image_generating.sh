# !/bin/bash

port_number=53102

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../image_generating.py \
 --output_dir "/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/anomal_source" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_weights "/home/dreamyou070/AnomalLora/result/bagel/caption_good_res_64_attnloss_1_down_dim_160/models/epoch-000014.safetensors" \
 --train_unet \
 --train_text_encoder \
 --obj_name "bagel" \
 --prompt_list "['random hole pattern','random crack pattern']"