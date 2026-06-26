_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn_20c.py',
    '../../_base_/datasets/voc0712_daniel_loc20_mi20_bkg20_asym40.py',
    '../../_base_/default_runtime.py',
]

env_cfg = dict(cudnn_benchmark=False)

custom_imports = dict(
    imports=[
        'custom_configs.models.shared2fc_bbox_head_oamil',
        'custom_configs.models.standard_roi_head_oamil',
        'custom_configs.hooks.oamil_epoch_injector_hook',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(type='OAMILEpochInjectorHook'),
]

# Production OA-MIL config (matches the paper's VOC hyperparameters).
# Full pipeline enabled: OA-IS pseudo + loss_oais + OA-IE iterative
# refinement. OA-IE activates only at epoch 9 (1-indexed); for fast
# iteration-level validation use exp_oamil_voc_sim40_oaie_early.py
# which overrides oaie_epoch=2.
model = dict(
    roi_head=dict(
        type='StandardRoIHeadOAMIL',
        bbox_head=dict(
            type='Shared2FCBBoxHeadOAMIL',
            num_classes=20,
            # OA-MIL params (paper VOC values)
            oamil_lambda=0.1,
            oais_flag=True,
            oais_epoch=2,
            oais_gamma=7.5,
            oais_theta=0.85,
            oaie_flag=True,
            oaie_num=4,
            oaie_coef=1.0,
            oaie_epoch=9,
            oaie_type='refine',
        ),
    ),
)

# Same schedule / optimizer as exp5_12x1[9]_baseline_danieldb_sim40.py
# so that loss values can be compared 1:1.
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[9],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=16)

# Fixed seed so the Step 1 numerical comparison against the baseline is
# reproducible. Override via --cfg-options randomness.seed=N if needed.
randomness = dict(seed=2025, deterministic=False)
