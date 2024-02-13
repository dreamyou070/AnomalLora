#! /bin/bash

class_name="cable_gland"
second_folder_name="down_1_basic"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name}