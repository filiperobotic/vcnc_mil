# exps coco 

#loc 20
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc20_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc20_vcnc_gate_e4off_nc120;

#loc 40
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc40_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc40_vcnc_gate_e4off_nc120;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc20_schedule_9mile_eoff_nc90.py'  --work-dir ./work_dirs/coco_loc40_vcnc_gate_e4off_nc90;

#loc 60
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc60_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc60_vcnc_gate_e4off_nc120;

#simetrico 20
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_sim20_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_sim20_vcnc_gate_e4off_nc120;

#simetrico 40
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_sim40_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_sim40_vcnc_gate_e4off_nc120;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_sim40_schedule_9mile_eoff_nc90.py'  --work-dir ./work_dirs/coco_sim40_vcnc_gate_e4off_nc90;

#simetrico 60
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_sim60_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_sim60_vcnc_gate_e4off_nc120;

#Assimetrico 20
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_asim20_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_asim20_vcnc_gate_e4off_nc120;

#Assimetrico 40
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_asim40_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_asim40_vcnc_gate_e4off_nc120;
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_asim40_schedule_9mile_eoff_nc90.py'  --work-dir ./work_dirs/coco_asim40_vcnc_gate_e4off_nc90;

#Loc + asim 20
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc20_asim20_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc20_asim20_vcnc_gate_e4off_nc120;

#Loc + asim 40
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc40_asim40_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc40_asim40_vcnc_gate_e4off_nc120;

# Loc + mi + bkg + sim 20
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc20_mi20_bkg20_sim20_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc20_mi20_bkg20_sim20_vcnc_gate_e4off_nc120;

# Loc + mi + bkg + sim 40
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc40_mi40_bkg40_sim40_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc40_mi40_bkg40_sim40_vcnc_gate_e4off_nc120;

# Loc + mi + bkg + asim 20
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc20_mi20_bkg20_asim20_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc20_mi20_bkg20_asim20_vcnc_gate_e4off_nc120;

# Loc + mi + bkg + asim 40
python3 tools/train.py './custom_configs/daniel_oamil/faster-rcnn_r50_fpn_1x_COCO_gateOAMIL_loc40_mi40_bkg40_asim40_schedule_9mile_eoff_nc120.py'  --work-dir ./work_dirs/coco_loc40_mi40_bkg40_asim40_vcnc_gate_e4off_nc120;







