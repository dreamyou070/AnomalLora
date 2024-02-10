#! /bin/bash

class_name="bagel"
dataset_cat="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${dataset_cat}"
sub_folder="1_64_down_total_normal_thred_0"


output_dir="metrics"

python ../evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "../../result/${class_name}/${sub_folder}/reconstruction" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3