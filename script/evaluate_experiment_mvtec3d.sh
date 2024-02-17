#! /bin/bash

class_name="carrot"
bench_mark="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
#sub_folder="9_up_2_anomal_pe_down_down_mahal_task_loss"
#sub_folder="10_up_2_anomal_pe_down_down_mahal_task_loss_only_local_attn"
#sub_folder="11_up_2_anomal_pe_down_down_mahal_task_loss_global_local_attn"
sub_folder="14_up_2_anomal_pe_down_focal_loss"
#sub_folder="15_up_2_anomal_pe_down_do_query_shuffle_loss"

output_dir="metrics"


python ../evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "../../result/${bench_mark}/${class_name}" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --sub_folder "${sub_folder}" \
     --pro_integration_limit 0.3
