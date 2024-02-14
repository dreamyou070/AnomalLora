# !/bin/bash

port_number=51544

obj_name='cable_gland'
trigger_word='cable'

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_with_positionembedding.py \
 --log_with wandb \
 --output_dir "../../result/${obj_name}/up_2_map_loss_image_classification_on_same_block" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' --beta_scale_factor 0.8 \
 --use_position_embedder --d_dim 320 --latent_res 64 --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' \
 --trigger_word "${trigger_word}" --obj_name "${obj_name}" --train_unet --train_text_encoder \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --start_epoch 0 --max_train_epochs 300 --anomal_only_on_object --unet_inchannels 4 --min_timestep 0 --max_timestep 1000 \
 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --do_map_loss \
 --do_classification --image_classification_layer "up_blocks_3_attentions_2_transformer_blocks_0_attn2"