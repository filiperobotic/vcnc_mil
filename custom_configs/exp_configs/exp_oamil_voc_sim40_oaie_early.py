# Fast-validation variant of the production OA-MIL config.
#
# Differs from exp_oamil_voc_sim40.py only in oaie_epoch: dropped from 9
# to 2 so that OA-IE iteration kicks in at epoch 2 (1-indexed). Lets the
# Step 4 wiring be validated in 3 epochs instead of 10.
#
# Production runs (paper hyperparameters) should still use
# exp_oamil_voc_sim40.py — this file is intentionally diagnostic only.

_base_ = ['./exp_oamil_voc_sim40.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            oaie_epoch=2,
        ),
    ),
)
