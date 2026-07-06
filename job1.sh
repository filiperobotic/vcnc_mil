python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_OAMIL_loc40_schedule_1x.py'  --work-dir ./work_dirs/oamil3x_daniel_sched1x_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_OAMIL_loc40_schedule_1x.py'  --work-dir ./work_dirs/oamil3x_daniel_sched1x_run2;

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_OAMIL_loc40_schedule_9mile.py'  --work-dir ./work_dirs/oamil3x_daniel_sched9mi_run1;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_voc_OAMIL_loc40_schedule_9mile.py'  --work-dir ./work_dirs/oamil3x_daniel_sched9mi_run2;

python3 tools/train.py './custom_configs/exp_configs/exp_oamil_filipe_voc_loc40_sched1x.py'  --work-dir ./work_dirs/oamil_filipe_sched1x_run1;
python3 tools/train.py './custom_configs/exp_configs/exp_oamil_filipe_voc_loc40_sched1x.py'  --work-dir ./work_dirs/oamil_filipe_sched1x_run2;

python3 tools/train.py './custom_configs/exp_configs/exp_oamil_filipe_voc_loc40_sched9mi.py'  --work-dir ./work_dirs/oamil_filipe_sched9mi_run1;
python3 tools/train.py './custom_configs/exp_configs/exp_oamil_filipe_voc_loc40_sched9mi.py'  --work-dir ./work_dirs/oamil_filipe_sched9mi_run2;
