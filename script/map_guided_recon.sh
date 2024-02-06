# !/bin/bash

port_number=59845
obj_name='bagel'
caption='good'

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../map_guided_recon.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_folder "../../result/${obj_name}/caption_good_res_64_general_anomal_source_partial_anomal_attnloss_0.1_anomal_th_max_anomal_sample_normal_loss_down_dim_160/models" \
 --data_path "../../../MyData/anomaly_detection/MVTec3D-AD/${obj_name}/test" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']" \
 --down_dim 160 \
 --prompt "${caption}"