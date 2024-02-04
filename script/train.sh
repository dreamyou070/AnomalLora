# !/bin/bash

port_number=53103
obj_name='bagel'

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../train.py \
 --output_dir "../../result/${obj_name}/down_64_task_1_dist_1_attn_0.01_normal_0.5" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' \
 --obj_name "${obj_name}" \
 --train_unet \
 --train_text_encoder \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']" \
 --do_task_loss --task_loss_weight 1.0 \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 0.01 \
 --normal_weight 0.5 \
 --num_epochs 30