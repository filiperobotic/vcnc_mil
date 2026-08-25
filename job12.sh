python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc20_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc20_vcnc_gate_e4off_nc120;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc20_schedule_9mile_eoff_nc90.py'  --work-dir ./work_dirs/coco_loc20_vcnc_gate_e4off_nc90;

python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc60_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc60_vcnc_gate_e4off_nc120;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc60_schedule_9mile_eoff_nc90.py'  --work-dir ./work_dirs/coco_loc20_vcnc_gate_e4off_nc90;


