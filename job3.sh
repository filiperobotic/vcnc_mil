python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_bkg40_mi40_sim40_schedule_9mile.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_sched9mi_loc40_bkg40_mi40_sim40_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_bkg40_mi40_sim40_schedule_9mile.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_sched9mi_loc40_bkg40_mi40_sim40_run2;

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_OAMIL_loc40_bkg40_mi40_sim40_schedule_9mile.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_sched9mi_loc40_bkg40_mi40_sim40_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_OAMIL_loc40_bkg40_mi40_sim40_schedule_9mile.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_sched9mi_loc40_bkg40_mi40_sim40_run2;


