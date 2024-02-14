# !/bin/bash

port_number=54414
obj_name='cable_gland'
trigger_word='cable'

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_only_hole.py \
 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
 --output_dir "../../result/${obj_name}/64_up_2_train_only_hole_normal_without_background" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --train_unet --train_text_encoder \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --start_epoch 0 \
 --max_train_epochs 300 \
 --num_repeat 1 \
 --anomal_only_on_object \
 --unet_inchannels 4 \
 --min_timestep 0 \
 --max_timestep 1000 \
 --truncating --latent_res 64  \
 --truncating 3