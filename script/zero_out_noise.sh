# !/bin/bash


port_number=59144
obj_name='bagel'
trigger_word='bagel'

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../zero_out_noise.py \
 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
 --output_dir "../../result/${obj_name}/6_caption_bagel_64_down_1_timestep_thred_0.1" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --train_unet --train_text_encoder \
 --network_dim 64 --network_alpha 4 \
 --trg_layer_list "['down_blocks_0_attentions_1_transformer_blocks_0_attn2']"  \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --max_train_epochs 300 --start_epoch 0 --num_repeat 1 --anomal_only_on_object --unet_inchannels 4 \
 --background_with_normal --truncating \
 --normal_without_init_noise --timestep_thred_ratio 0.1