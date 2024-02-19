#! /bin/bash

class_name="carrot"
second_folder_name="20_4_1_do_anomal_sample_do_attn_loss_do_map_loss"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name}