# !/bin/bash

port_number=59202
obj_name='bagel'
caption='bagel'


accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction_final.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_folder "../../result/${obj_name}/6_caption_bagel_64_down_harsh_timestep/models" \
 --object_detector_weight "../../result/${obj_name}/object_detector_experiments/object_detector_1/models/epoch-000100.safetensors" \
 --data_path "../../../MyData/anomaly_detection/MVTec3D-AD/${obj_name}/test" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']" \
 --network_dim 64 --network_alpha 4 \
 --down_dim 320 \
 --unet_inchannels 4 \
 --prompt "${caption}" --truncating