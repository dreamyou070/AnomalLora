# !/bin/bash

port_number=50011

obj_name='bottle'
trigger_word='bottle'
bench_mark='MVTec'
anomal_source_path="../../../MyData/anomal_source"
# --do_down_dim_mahal_loss --down_dim 100 \

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/1_up_2_anomal_pe_down" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --trigger_word "${trigger_word}" --obj_name "${obj_name}" \
 --anomal_source_path ${anomal_source_path} \
 --beta_scale_factor 0.8 --anomal_only_on_object --anomal_source_path "${anomal_source_path}" \
 --train_unet --train_text_encoder \
 --use_position_embedder --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' --d_dim 320 --latent_res 64  \
 --start_epoch 0 --max_train_epochs 50 \
 --do_dist_loss --dist_loss_weight 1.0 \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --do_map_loss