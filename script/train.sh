# !/bin/bash

port_number=50015

obj_name='carrot'
trigger_word='carrot'
bench_mark='MVTec3D-AD'
anomal_source_path="../../../MyData/anomal_source"
#  \
#
# --use_focal_loss

accelerate launch --config_file ../../../gpu_config/gpu_0_1_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/15_up_2_anomal_pe_down_do_query_shuffle_loss" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --beta_scale_factor 0.8 --anomal_source_path "${anomal_source_path}" --anomal_only_on_object \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --trigger_word "${trigger_word}" --obj_name "${obj_name}" --anomal_only_on_object \
 --use_position_embedder --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' --d_dim 320 --latent_res 64 \
 --do_dist_loss --dist_loss_weight 1.0 --do_down_dim_mahal_loss --down_dim 100 \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --do_map_loss --do_query_shuffle_loss