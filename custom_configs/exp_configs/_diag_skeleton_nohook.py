# Diagnostic config: identical to exp_oamil_voc_sim40.py EXCEPT it removes
# OAMILEpochInjectorHook from custom_hooks. The OA-MIL classes are still in
# the model. Purpose: isolate whether OAMILEpochInjectorHook itself perturbs
# the run.

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn_20c.py',
    '../_base_/datasets/voc0712_daniel_sim40.py',
    '../_base_/default_runtime.py',
]

env_cfg = dict(cudnn_benchmark=False)

custom_imports = dict(
    imports=[
        'custom_configs.models.shared2fc_bbox_head_oamil',
        'custom_configs.models.standard_roi_head_oamil',
    ],
    allow_failed_imports=False,
)

custom_hooks = []  # ← OAMILEpochInjectorHook removed for this diagnostic

model = dict(
    roi_head=dict(
        type='StandardRoIHeadOAMIL',
        bbox_head=dict(
            type='Shared2FCBBoxHeadOAMIL',
            num_classes=20,
            oamil_lambda=0.0,
            oais_flag=False,
            oais_epoch=2,
            oais_gamma=7.5,
            oais_theta=0.85,
            oaie_flag=False,
            oaie_num=4,
            oaie_coef=1.0,
            oaie_epoch=9,
            oaie_type='refine',
        ),
    ),
)

max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
         milestones=[9], gamma=0.1),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=16)

randomness = dict(seed=2025, deterministic=False)
