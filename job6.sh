#exps2

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-on) - asym40%

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_asym40_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_asym40_run1v2;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_asym40_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_asym40_run2v2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-on) - loc40%

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_loc40_run1v2;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_loc40_run2v2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-on) - sim40%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_sim40_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_sim40_run1v2;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_sim40_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_sim40_run2v2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-on) - loc20%

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc20_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_loc20_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc20_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_loc20_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-on) - sim20%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_sim20_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_sim20_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_sim20_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_sim20_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-on) - asym40%

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_asym20_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_asym20_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_asym20_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_asym20_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - loc 40%

python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc40_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc40_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - sim 40%

python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_sim40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_simm40_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_sim40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_simm40_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - asym 40%

python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_asym40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_asym40_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_asym40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_asym40_run2;
