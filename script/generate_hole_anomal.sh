# !/bin/bash

port_number=54412
obj_name='cable_gland'
trigger_word='cable'

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train_pixel_anomal.py \
 --data_path '../../../MyData/anomaly_detection/MVTec3D-AD' \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --anomal_only_on_object \
 --unet_inchannels 4 \
 --min_timestep 0 \
 --max_timestep 1000 \
 --truncating --latent_res 64 \
 --total_normal_thred 0.3 \
 --perlin_max_scale 6 \
 --kernel_size 9