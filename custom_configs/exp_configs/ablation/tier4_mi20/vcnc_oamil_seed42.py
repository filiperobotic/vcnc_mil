_base_ = ['../../ablation_intermediates/exp_vcnc_gate_oamil_full_mi20.py']

train_cfg = dict(max_epochs=12, val_interval=1)
randomness = dict(seed=42, deterministic=True)
default_hooks = dict(checkpoint=None)
