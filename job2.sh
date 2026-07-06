python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_schedule_9mile.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_sched9mi_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_schedule_9mile.py'  --work-dir ./work_dirs/vcnc_gate_oamil3x_daniel_sched9mi_run2;

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_schedule_1x.py'  --work-dir ./work_dirs/vcnc_gate_oami_daniel_sched1x_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_VCNC_gateOAMIL_loc40_schedule_1x.py'  --work-dir ./work_dirs/vcnc_gate_oami_daniel_sched1x_run2;

