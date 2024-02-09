# !/bin/bash

port_number=59302
obj_name='bagel'
caption='bagel'

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../generation_test.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_folder "/home/dreamyou070/Lora/OODLora/result/MVTec3D-AD_experiment/bagel/lora_training/normal/object_detection/models" \
 --object_detector_weight "../../result/${obj_name}/object_detector/models/epoch-000100.safetensors" \
 --data_path "../../../MyData/anomaly_detection/MVTec3D-AD/${obj_name}/test" \
 --obj_name "${obj_name}" \
 --down_dim 320 \
 --prompt "${caption}"
 #--trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
