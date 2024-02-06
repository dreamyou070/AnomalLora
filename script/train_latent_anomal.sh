# !/bin/bash

port_number=58510
obj_name='bagel'
trigger_word='good'

output_dir="../../result/${obj_name}/caption_good_res_64_attnloss_1_pixel_anomal"
network_weights="../../result/${obj_name}/caption_good_res_64_attnloss_1_pixel_anomal/models/epoch-000030.safetensors"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_config \
 --main_process_port $port_number ../train_latent_anomal.py \
 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
 --output_dir ${output_dir} \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' \
 --obj_name "${obj_name}" \
 --train_unet --train_text_encoder \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']" \
 --num_epochs 100 \
 --trigger_word "${trigger_word}" \
 --do_task_loss --task_loss_weight 1.0 --do_cls_train \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 1.0 --normal_weight 1 \
 --do_anomal_sample_normal_loss \
 --down_dim 320 \
 --network_weights ${network_weights} \
 --start_epoch 30 \
 --num_repeat 1