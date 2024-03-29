# !/bin/bash

port_number=52413
obj_name='carrot'
caption='carrot'
folder_name="0_reference_check_do_attn_loss_second_attn_gen_code_with_background_img"
bench_mark="MVTec3D-AD"
position_embedding_layer="down_blocks_0_attentions_0_transformer_blocks_0_attn1"
# --do_local_self_attn --window_size 8
#--data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 --network_folder "../../result/${bench_mark}/${obj_name}/${folder_name}/models" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}" --prompt "${caption}" \
 --latent_res 64 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --d_dim 320 --use_position_embedder --position_embedding_layer ${position_embedding_layer} \
 --threds [0.5]