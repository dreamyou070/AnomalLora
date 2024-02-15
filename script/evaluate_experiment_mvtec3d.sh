#! /bin/bash

class_name="carrot"
bench_mark="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
sub_folder="up_2_not_anomal_hole_act_deact_do_down_dim_mahal_loss"

output_dir="metrics"


python ../evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "../../result/${bench_mark}/${class_name}/${sub_folder}/reconstruction" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3