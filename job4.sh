for scen_path in tier3_loc20 tier2_asym40 tier3_sym60 tier4_mi40 tier4_bkg40 tier1_loc20_asym20; do
    scen_name=$(echo $scen_path | sed 's/^tier[0-9]*_//')
    python tools/train.py \
        custom_configs/exp_configs/ablation/${scen_path}/vcnc_oamil_seed2025.py \
        --work-dir work_dirs/diag_concord_${scen_name} \
        --cfg-options \
            train_cfg.max_epochs=3 \
            train_cfg.val_interval=999 \
            default_hooks.logger.interval=500 \
        2>&1 | tee work_dirs/diag_concord_${scen_name}.stdout
done
