# !/bin/bash

port_number=50015

obj_name='carrot'
trigger_word='carrot'
bench_mark='MVTec3D-AD'

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../train_student_lora.py \
 --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/train_student_lora" \
 --network_weights "../../result/${obj_name}/up_2_not_anomal_hole_act_deact/models/epoch-000016.safetensors" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --beta_scale_factor 0.8 \
 --trigger_word "${trigger_word}" --obj_name "${obj_name}" --train_unet --train_text_encoder \
 --use_position_embedder --position_embedding_layer 'unet' \
 --d_dim 320 --latent_res 64 \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --start_epoch 0 --max_train_epochs 300 \
 --unet_inchannels 4 --min_timestep 0 --max_timestep 1000 \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --do_map_loss --use_focal_loss --down_dim 100