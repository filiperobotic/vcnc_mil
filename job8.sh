python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_asym20_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_asym20_run2;


#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - loc60%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc60_schedule_9mile_eoff.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_loc60_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc60_schedule_9mile_eoff.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_loc60_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - sim60%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_sim60_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_sim60_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_sim60_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_sim60_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-on) - loc60%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc60_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_loc60_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc60_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_loc60_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-on) - sim60%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_sim60_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_sim60_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_sim60_schedule_9mile_e4on.py'  --work-dir ./work_dirs/vcnc_gate_e4on_oamil3x_daniel_sched9mi_sim60_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - loc 20%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc20_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc20_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - sim 20%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_sim20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_sim20_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_sim20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_sim20_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - asym 20%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_asym20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_asym20_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_asym20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_asym20_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - loc 60%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc60_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc60_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc60_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc60_run2;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - sim 60%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_sim60_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_sim60_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_sim60_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_sim60_run2;

#vcnc_kmeans30_confusion_aware_gatefilter_spatial_fixed loc 20%
python3 tools/train.py './custom_configs/faster_rcnn/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc20_fixed.py'  --work-dir ./work_dirs/vcnc_gate_loc20_run1;
python3 tools/train.py './custom_configs/faster_rcnn/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc20_fixed.py'  --work-dir ./work_dirs/vcnc_gate_loc20_run2;

#vcnc_kmeans30_confusion_aware_gatefilter_spatial_fixed loc 40%
python3 tools/train.py './custom_configs/faster_rcnn/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc40_fixed.py'  --work-dir ./work_dirs/vcnc_gate_loc40_run1;
python3 tools/train.py './custom_configs/faster_rcnn/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc40_fixed.py'  --work-dir ./work_dirs/vcnc_gate_loc40_run2;

