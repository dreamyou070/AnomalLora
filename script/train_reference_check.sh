# !/bin/bash

port_number=50001

obj_name='carrot'
trigger_word='carrot'
bench_mark='MVTec3D-AD'
anomal_source_path="../../../MyData/anomal_source"
#--do_dist_loss
# --attention_loss_check_third
# --do_map_loss
accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_reference_check.py --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/0_reference_check_do_attn_loss_second_attn_gen_code_with_background_img" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --beta_scale_factor 0.8 --anomal_source_path "${anomal_source_path}" --anomal_only_on_object \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --trigger_word "${trigger_word}" --obj_name "${obj_name}" --anomal_only_on_object \
 --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' --d_dim 320 --latent_res 64 \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --do_anomal_sample --do_holed_sample --do_normal_sample \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1.0 --use_small_anomal