#! /bin/bash

class_name="cable_gland"
second_folder_name="up_2_not_anomal_hole_act_deact"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name}