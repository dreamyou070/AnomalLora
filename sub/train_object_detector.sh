# !/bin/bash

port_number=53822
obj_name='bagel'
trigger_word='bagel'

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_object_detector.py \
 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
 --output_dir "../../result/${obj_name}/object_detector_experiments/object_detector_20240209" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --train_unet --train_text_encoder \
 --network_dim 64 --network_alpha 4 \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --do_task_loss --task_loss_weight 1.0 --do_dist_loss --dist_loss_weight 1.0 \
 --do_attn_loss --attn_loss_weight 1.0 --do_cls_train --normal_weight 1 \
 --max_train_epochs 30 --start_epoch 0 --num_repeat 1 \
 --unet_inchannels 4
 #  --dist_loss_weight 1.0 --do_cls_train --num_epochs 30