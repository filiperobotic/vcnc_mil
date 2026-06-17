from typing import Optional

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead


@MODELS.register_module()
class Shared2FCBBoxHeadOAMIL(Shared2FCBBoxHead):
    """OA-MIL variant of Shared2FCBBoxHead.

    Stores OA-MIL hyperparameters and an ``_epoch`` attribute (updated by
    ``OAMILEpochInjectorHook``). When ``pseudo_bbox_targets`` is passed to
    ``loss``, it replaces ``bbox_targets`` at positive indices via a
    clone-and-replace, then delegates to ``super().loss`` — so the
    classification path stays bit-identical to the baseline and the only
    visible effect is on ``loss_bbox`` (regression against pseudo-GT
    instead of noisy GT).
    """

    def __init__(self,
                 oamil_lambda: float = 0.0,
                 oais_flag: bool = False,
                 oais_epoch: int = 12,
                 oais_gamma: float = 7.5,
                 oais_theta: float = 0.0,
                 oaie_flag: bool = False,
                 oaie_num: int = 0,
                 oaie_coef: float = 0.0,
                 oaie_epoch: int = 12,
                 oaie_type: str = 'refine',
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.oamil_lambda = oamil_lambda
        self.oais_flag = oais_flag
        self.oais_epoch = oais_epoch
        self.oais_gamma = oais_gamma
        self.oais_theta = oais_theta
        self.oaie_flag = oaie_flag
        self.oaie_num = oaie_num
        self.oaie_coef = oaie_coef
        self.oaie_epoch = oaie_epoch
        self.oaie_type = oaie_type
        # Updated by OAMILEpochInjectorHook before each train epoch.
        # Fallback 0 keeps OA-IS/OA-IE gated off if the hook isn't registered.
        self._epoch = 0

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None,
             pseudo_bbox_targets: Optional[Tensor] = None) -> dict:
        if pseudo_bbox_targets is None:
            return super().loss(
                cls_score, bbox_pred, rois,
                labels, label_weights, bbox_targets, bbox_weights,
                reduction_override=reduction_override)

        # OA-IS path: swap regression targets at positive indices.
        # pseudo_bbox_targets is expected with shape (num_pos, 4) — already
        # filtered by the same pos_mask the parent will recompute below.
        bg_class_ind = self.num_classes
        pos_mask = (labels >= 0) & (labels < bg_class_ind)
        assert pseudo_bbox_targets.shape == bbox_targets[pos_mask].shape, (
            'pseudo_bbox_targets shape mismatch: '
            f'{tuple(pseudo_bbox_targets.shape)} vs '
            f'{tuple(bbox_targets[pos_mask].shape)}')

        patched_targets = bbox_targets.clone()
        patched_targets[pos_mask] = pseudo_bbox_targets

        return super().loss(
            cls_score, bbox_pred, rois,
            labels, label_weights, patched_targets, bbox_weights,
            reduction_override=reduction_override)
