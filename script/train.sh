# !/bin/bash

port_number=53102

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
  --main_process_port $port_number ../train.py \
 --output_dir ../../result/20240204 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors