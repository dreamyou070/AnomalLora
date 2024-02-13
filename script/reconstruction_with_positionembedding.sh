# !/bin/bash
#

port_number=51010
obj_name='cable_gland'
caption='cable'
# --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1'
#

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction_with_positionembedding.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_folder "../../result/${obj_name}/up_2_basic_without_total_normal_pe_beta_scale_0.4/models" \
 --data_path "../../../MyData/anomaly_detection/MVTec3D-AD/${obj_name}/test" \
 --obj_name "${obj_name}" --network_dim 64 --network_alpha 4 --unet_inchannels 4 --prompt "${caption}" --latent_res 64 \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --d_dim 320 --use_position_embedder --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1'