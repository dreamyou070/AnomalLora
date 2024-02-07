# !/bin/bash

port_number=58555
obj_name='bagel'
trigger_word='bagel'
output_dir="../../result/${obj_name}/latent_caption_bagel_down_dim_320_more_generalize_attn_loss_0.001"
#network_weights="../../result/${obj_name}/caption_bagel_down_dim_320_more_generalize/models//epoch-000011.safetensors"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_config \
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
 --do_task_loss --task_loss_weight 1.0 --down_dim 320 --num_repeat 1 \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_cls_train --do_attn_loss --attn_loss_weight 0.001 --normal_weight 1 \
 #--do_anomal_sample_normal_loss \
 #--start_epoch 11 \
 #--more_generalize \
 #--network_weights ${network_weights}