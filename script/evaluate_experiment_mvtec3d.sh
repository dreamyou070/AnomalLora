#! /bin/bash

class_name="carrot"
bench_mark="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
sub_folder="8_up_2_anomal_pe_unet_down_mahal"

output_dir="metrics"


python ../evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "../../result/${bench_mark}/${class_name}/${sub_folder}/reconstruction_eval" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3