_base_ = [
    './faster-rcnn_r50_fpn_oamil.py',
    '../_base_/datasets/voc0712_daniel_loc40.py',
    #'../_base_/schedules/schedule_1x.py', 
    '../../configs/_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='NOD')),
]

visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')