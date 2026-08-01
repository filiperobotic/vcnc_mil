#exps 4

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - mi20%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_mi20_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_mi20;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - bkg20%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_bkg20_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_bkg20;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - loc20 asym20%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc20_asym20_schedule_9mile_eoff.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_loc20asym20;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - loc mi bkg sim20%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc20_mi20_bkg20_sim20_schedule_9mile_eoff.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel9mi_loc_mi_bkg_sim20;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - loc mi bkg asym20%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc20_mi20_bkg20_asym20_schedule_9mile_eoff.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel9mi_loc_mi_bkg_asym20;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - mi40%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_mi40_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_mi40;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - bkg40%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_bkg40_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_bkg40;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - loc40 asym40%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_asym40_schedule_9mile_eoff.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel9mi_loc40_asym40;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - loc mi bkg asym40%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_mi40_bkg40_asym40_schedule_9mile_eoff.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel9mi_loc_mi_bkg_asym40;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - mi60%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_mi60_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_mi60;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - bkg60%
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_bkg60_schedule_9mile_e4off.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel_sched9mi_bkg60;


#vcnc_kmeans30_confusion_aware_gatefilter_spatial_fixed loc 20%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc20_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc20_run2;


#vcnc_kmeans30_confusion_aware_gatefilter_spatial_fixed loc 40%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc40_run1;
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc40_run2;


#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - mi 20%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_mi20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_mi20;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - bkg 20%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_bkg20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_bkg20;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - loc mi bkg asym 20%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc20_mi20_bkg20_asym20_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc_mi_bkg_asym20;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - mi 40%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_mi40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_mi40;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - bkg 40%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_bkg40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_bkg40;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - loc asym 40%

python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc40_asym40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc40_asym40;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - loc mi bkg sim 40%

python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_loc40_mi40_bk40_sim40_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_loc_mi_bkg_sim40;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - mi 60%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_mi60_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_mi60;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + KL-Loss - bkg 60%
python3 tools/train.py './custom_configs/daniel_klloss/exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_bkg60_fixed_KL.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_KL_sched9mi_bkg60;

#vcnc_kmeans30_confustion_aware_gatefilter_spatial_fixed + oamil_daniel_sched9mi (e4-off) - loc mi bkg sim 60%

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc60_mi60_bkg60_sim60_schedule_9mile_eoff.py'  --work-dir ./work_dirs/vcnc_gate_e4off_oamil3x_daniel9mi_loc_mi_bkg_sim60;




