_base_ = [
    './faster-rcnn_r50_fpn_oamil.py',
    '../_base_/datasets/voc0712_daniel_loc20.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]