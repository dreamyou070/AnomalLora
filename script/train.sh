# !/bin/bash

port_number=53877
obj_name='bagel'
caption='bagel'

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../train.py \
 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
 --output_dir "../../result/${obj_name}/caption_${caption}_res_64_general_anomal_source_partial_anomal" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' \
 --obj_name "${obj_name}" \
 --train_unet \
 --train_text_encoder \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']" \
 --do_task_loss --task_loss_weight 1.0 \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 0.1 \
 --normal_weight 1 \
 --do_synthetic_anomaly \
 --num_epochs 30