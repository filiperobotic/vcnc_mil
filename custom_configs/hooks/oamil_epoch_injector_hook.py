from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class OAMILEpochInjectorHook(Hook):
    """Inject the current epoch into ``roi_head.bbox_head._epoch``.

    OA-MIL gates OA-IS / OA-IE on ``_epoch + 1 >= oais_epoch / oaie_epoch``.
    Without this hook the attribute stays at its ``__init__`` value (0),
    so the gates never open. The hook is a no-op for any model whose
    bbox_head doesn't carry an ``_epoch`` attribute, so adding it to
    ``custom_hooks`` of a non-OA-MIL config is safe.
    """

    priority = 'NORMAL'

    def before_train_epoch(self, runner) -> None:
        model = runner.model
        # Unwrap DDP / MMDistributedDataParallel
        if hasattr(model, 'module'):
            model = model.module
        bbox_head = getattr(getattr(model, 'roi_head', None), 'bbox_head', None)
        if bbox_head is not None and hasattr(bbox_head, '_epoch'):
            bbox_head._epoch = runner.epoch
