#! /bin/bash

class_name="cable_gland"
second_folder_name="up_2_basic_without_total_normal_pe_beta_scale_0.8"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name}