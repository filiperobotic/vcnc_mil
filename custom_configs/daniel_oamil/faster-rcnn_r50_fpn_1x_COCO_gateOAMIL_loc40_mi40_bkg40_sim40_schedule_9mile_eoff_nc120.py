_base_ = [
    './faster-rcnn_r50_fpn_oamil_coco.py',
    '../_base_/datasets/coco_detection_loc40_mi40_bkg40_sim40.py',
    #'../_base_/schedules/schedule_1x.py', 
    # '../../configs/_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='NOD')),
]

visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')


custom_imports = dict(
    imports=[
        'custom_configs.hooks.vcnc_kmeans_confusion_aware_spatial_hook',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(
        type='VCNCKMeansConfusionAwareHook',
        warmup_epochs=1,
        num_classes=80,

        enable_confidence_relabel=False,           # E1 OFF (FIXED)
        relabel_confidence_threshold=0.9,

        enable_clustering_relabel=True,            # E2 ON
        #n_clusters=30,
        n_clusters=120,
        use_softmax_as_embedding=True,
        progressive_epochs=4,                       # FIXED — gate at epoch 5

        early_anchor_gmm_threshold=0.15,
        early_anchor_pred_agreement=0.85,
        early_anchor_confidence=0.9,
        early_suspect_gmm_threshold=0.8,
        early_similarity_threshold=0.7,
        early_cluster_consensus=0.85,

        anchor_gmm_threshold=0.4,
        anchor_pred_agreement=0.6,
        anchor_confidence=0.7,
        suspect_gmm_threshold=0.5,
        similarity_threshold=0.4,
        cluster_consensus=0.6,

        confusion_gate_min_samples=50,
        confusion_gate_mad_factor=3.0,
        confusion_gate_ratio_factor=10.0,
        confusion_gate_aggressive_only=True,        # FIXED
        confusion_gate_action='filter',             # FIXED
        enable_confusion_gate=True,

        enable_spatial_refinement=True,             # E3 ON
        spatial_difficulty_threshold=0.5,

        enable_gmm_filter=False,                    # E4 OFF (FIXED)
        gmm_components=4,
        filter_gmm_threshold=0.7,

        iou_assigner=0.5,
        reload_dataset=True,
        debug=True,
    ),
]



# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[9],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
