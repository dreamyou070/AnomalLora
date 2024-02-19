#! /bin/bash

class_name="carrot"
second_folder_name="21_1_1_do_anomal_do_holed_sample_do_attn_loss"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name}