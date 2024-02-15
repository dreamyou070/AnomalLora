# !/bin/bash
port_number=50004

obj_name='bottle'
trigger_word='bottle'
bench_mark='MVTec3D'

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_config \
 --main_process_port $port_number ../train_with_positionembedding.py \
 --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/up_2_not_anomal_hole_act_deact_do_down_dim_mahal_loss_map_loss_with_focal_loss" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --beta_scale_factor 0.8 \
 --use_position_embedder --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' \
 --d_dim 320 --latent_res 64 --position_embedding_layer 'unet' \
 --trigger_word "${trigger_word}" --obj_name "${obj_name}" --train_unet --train_text_encoder \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --start_epoch 0 --max_train_epochs 300 --anomal_only_on_object --unet_inchannels 4 --min_timestep 0 --max_timestep 1000 \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --do_map_loss --use_focal_loss --down_dim 100