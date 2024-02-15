#! /bin/bash

class_name="carrot"
second_folder_name="up_2_not_anomal_hole_act_deact_do_down_dim_mahal_loss_map_loss_with_focal_loss_on_unet"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name}