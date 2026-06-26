# Production combo: VCNCKMeansConfusionAwareHook (gate, FIXED) +
# full OA-MIL pipeline with paper-equivalent VOC hyperparameters.
#
# Trains for 12 epochs. OA-MIL gates open per the production
# exp_oamil_voc_asym20.py schedule (OA-IS at epoch 2, OA-IE at epoch 9).
# VCNC E2/E3 fire from epoch 2 (after warmup_epochs=1); confusion gate
# fires first at epoch 5 (progressive_epochs=4 + aggressive_only=True →
# epoch > 4).
#
# === DATASET ===
# Inherits dataset from exp_oamil_voc_asym20.py (variant: asym20).

_base_ = ['./exp_oamil_voc_asym20.py']

custom_imports = dict(
    imports=[
        'custom_configs.models.shared2fc_bbox_head_oamil',
        'custom_configs.models.standard_roi_head_oamil',
        'custom_configs.hooks.oamil_epoch_injector_hook',
        'custom_configs.hooks.vcnc_kmeans_confusion_aware_spatial_hook',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(type='OAMILEpochInjectorHook'),
    dict(
        type='VCNCKMeansConfusionAwareHook',
        warmup_epochs=1,
        num_classes=20,

        enable_confidence_relabel=False,           # E1 OFF (FIXED)
        relabel_confidence_threshold=0.9,

        enable_clustering_relabel=True,            # E2 ON
        n_clusters=30,
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
