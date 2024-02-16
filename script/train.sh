# !/bin/bash

port_number=50002

obj_name='bottle'
trigger_word='bottle'
bench_mark='MVTec'
anomal_source_path="../../../MyData/anomal_source"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train.py \
 --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/1_basic_pe_unet" \
 --bench_mark "${bench_mark}" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --beta_scale_factor 0.8 \
 --anomal_source_path "${anomal_source_path}" \
 --trigger_word "${trigger_word}" --obj_name "${obj_name}" --train_unet --train_text_encoder \
 --use_position_embedder --position_embedding_layer 'unet' --d_dim 320 --latent_res 64  \
 --start_epoch 0 --max_train_epochs 300 \
 --do_dist_loss --dist_loss_weight 1.0 \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --do_map_loss