# Smoke test combining VCNCKMeansConfusionAwareHook (gate, FIXED config)
# with the full OA-MIL pipeline (OA-IS + loss_oais + OA-IE early).
#
# Designed to run end-to-end in ~3 epochs:
#   - OA-MIL fires at epoch 2 (oais_epoch=2, oaie_epoch=2 via base config).
#   - VCNC E2/E3 fire from epoch 2 (after warmup_epochs=1).
#   - VCNC confusion gate fires first at epoch 3 (progressive_epochs=2 +
#     confusion_gate_aggressive_only=True → epoch > 2 needed).
#
# === DATASET ===
# By default this config inherits sim40 (from
# exp_oamil_voc_sim40_oaie_early.py → exp_oamil_voc_sim40.py →
# ../_base_/datasets/voc0712_daniel_sim40.py).
#
# To swap noise variant:
#   (a) Append a dataset entry to _base_ (last wins, e.g.):
#         _base_ = ['./exp_oamil_voc_sim40_oaie_early.py',
#                   '../_base_/datasets/voc0712_daniel_loc40.py']
#   (b) CLI override:
#         python tools/train.py THIS_CONFIG \
#             --cfg-options \
#                 train_dataloader.dataset.ann_subdir='OTHER/PATH/' \
#                 val_dataloader.dataset.ann_subdir='OTHER/PATH/'

_base_ = ['./exp_oamil_voc_sim40_oaie_early.py']

# Override custom_imports to include the gate hook on top of the OA-MIL
# pieces from the base. mmengine REPLACES the dict, so list all imports.
custom_imports = dict(
    imports=[
        'custom_configs.models.shared2fc_bbox_head_oamil',
        'custom_configs.models.standard_roi_head_oamil',
        'custom_configs.hooks.oamil_epoch_injector_hook',
        'custom_configs.hooks.vcnc_kmeans_confusion_aware_spatial_hook',
    ],
    allow_failed_imports=False,
)

# Both hooks use priority='NORMAL' and before_train_epoch. There is no
# functional dependency between them (the OA-MIL injector only writes
# bbox_head._epoch; the VCNC hook only touches dataset annotations).
# Ordered OA-MIL first because it's a constant-time attribute write —
# the VCNC clustering can take seconds.
custom_hooks = [
    dict(type='OAMILEpochInjectorHook'),
    dict(
        type='VCNCKMeansConfusionAwareHook',
        warmup_epochs=1,
        num_classes=20,

        # E1 OFF (FIXED)
        enable_confidence_relabel=False,
        relabel_confidence_threshold=0.9,  # ignored when E1 is off

        # E2 ON
        enable_clustering_relabel=True,
        n_clusters=30,
        use_softmax_as_embedding=True,
        progressive_epochs=2,  # SMOKE — gate first fires at epoch 3

        # Conservative criteria (progressive phase, epochs <= 2)
        early_anchor_gmm_threshold=0.15,
        early_anchor_pred_agreement=0.85,
        early_anchor_confidence=0.9,
        early_suspect_gmm_threshold=0.8,
        early_similarity_threshold=0.7,
        early_cluster_consensus=0.85,

        # Aggressive criteria (after progressive phase)
        anchor_gmm_threshold=0.4,
        anchor_pred_agreement=0.6,
        anchor_confidence=0.7,
        suspect_gmm_threshold=0.5,
        similarity_threshold=0.4,
        cluster_consensus=0.6,

        # GATE (statistical, non-tunable)
        confusion_gate_min_samples=50,
        confusion_gate_mad_factor=3.0,
        confusion_gate_ratio_factor=10.0,
        confusion_gate_aggressive_only=True,  # FIXED
        confusion_gate_action='filter',       # FIXED
        enable_confusion_gate=True,

        # E3 ON
        enable_spatial_refinement=True,
        spatial_difficulty_threshold=0.5,

        # E4 OFF (FIXED)
        enable_gmm_filter=False,
        gmm_components=4,
        filter_gmm_threshold=0.7,

        iou_assigner=0.5,
        reload_dataset=True,
        debug=True,
    ),
]
