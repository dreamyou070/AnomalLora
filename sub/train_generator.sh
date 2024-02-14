# !/bin/bash

port_number=51101
obj_name='bagel'
trigger_word='bagel'

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_generator.py \
 --log_with wandb --wandb_api_key 3a3bc2f629692fa154b9274a5bbe5881d47245dc \
 --output_dir "../../result/${obj_name}/object_detector_experiments/just_train_20240209" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --train_unet --train_text_encoder \
 --network_dim 64 --network_alpha 4 \
 --do_task_loss --max_train_epochs 300 --start_epoch 0 --num_repeat 1 --unet_inchannels 4