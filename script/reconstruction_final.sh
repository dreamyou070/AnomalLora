# !/bin/bash

port_number=59207
obj_name='bagel'
caption='bagel'

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction_final.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_folder "../../result/${obj_name}/7_caption_bagel_1_with_taskloss_dist_loss_background_with_normal_more_anomal_srcs/models" \
 --object_detector_weight "../../result/${obj_name}/object_detector_experiments/object_detector_1/models/epoch-000100.safetensors" \
 --data_path "../../../MyData/anomaly_detection/MVTec3D-AD/${obj_name}/test" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --down_dim 320 \
 --prompt "${caption}"