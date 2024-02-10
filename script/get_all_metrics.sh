#! /bin/bash

class_name="bagel"
second_folder_name="1_64_down_total_normal_thred_0"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name}
