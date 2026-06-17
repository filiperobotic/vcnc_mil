# Diagnostic config: identical to exp_oamil_voc_sim40.py EXCEPT it uses the
# stock StandardRoIHead / Shared2FCBBoxHead instead of the OA-MIL classes and
# does NOT register OAMILEpochInjectorHook.
#
# Purpose: isolate the OA-MIL code change. If runs of this config and
# exp_oamil_voc_sim40.py (with the same seed + deterministic=True) produce
# bit-identical losses, the Step 1 skeleton is numerically transparent and
# the original baseline-vs-skeleton divergence was caused by wandb /
# custom_hooks asymmetry in the original baseline config.

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn_20c.py',
    '../_base_/datasets/voc0712_daniel_sim40.py',
    '../_base_/default_runtime.py',
]

env_cfg = dict(cudnn_benchmark=False)

# No custom_imports, no custom_hooks — keep the environment matched to the
# skeleton's environment minus only the OA-MIL pieces.
custom_hooks = []

model = dict(
    roi_head=dict(
        # Note: NOT StandardRoIHeadOAMIL
        bbox_head=dict(num_classes=20),
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
