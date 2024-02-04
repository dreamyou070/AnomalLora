# !/bin/bash

port_number=53102
obj_name='bagel'

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction.py \
 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_folder "../../result/${obj_name}/caption_bagel_res_64_train/models" \
 --test_img_folder "../../../MyData/anomaly_detection/MVTec3D-AD/${obj_name}/test" \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD/test' \
 --obj_name "${obj_name}" \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']"