from typing import List, Optional, Tuple

import torch
from torch import Tensor
from mmengine.logging import MMLogger

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.task_modules.samplers import SamplingResult


@MODELS.register_module()
class StandardRoIHeadOAMIL(StandardRoIHead):
    """OA-MIL variant of StandardRoIHead.

    Step 4: implements the full OA-MIL pipeline — OA-IS pseudo targets,
    MIL ``loss_oais`` aggregation, AND OA-IE iterative refinement. All
    three sub-features have independent epoch gates on top of their
    flags, allowing turning each on/off via the config alone.

    Activation gates:
      - OA-IS (``pseudo_bbox_targets``): ``oais_flag`` AND
        ``_epoch + 1 >= oais_epoch``.
      - ``loss_oais``: ``oamil_lambda > 0`` AND
        ``_epoch + 1 >= oais_epoch``.
      - OA-IE iteration: ``oaie_flag`` AND
        ``_epoch + 1 >= oaie_epoch`` AND ``loss_oais`` active (extra
        iterations are useless when no loss consumes them).

    All sub-pipelines share a single instance-selection method that
    walks the iteration loop once.
    """

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        cls_reg_targets = self.bbox_head.get_targets(
            sampling_results, self.train_cfg)

        loss_oais_dict, pseudo_bbox_targets = self._oamil(
            bbox_results, rois, cls_reg_targets, x)

        labels, label_weights, bbox_targets, bbox_weights = cls_reg_targets
        losses = self.bbox_head.loss(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            pseudo_bbox_targets=pseudo_bbox_targets,
        )
        losses.update(loss_oais_dict)

        bbox_results.update(loss_bbox=losses)
        return bbox_results

    def _oamil(self, bbox_results: dict, rois: Tensor,
               cls_reg_targets: Tuple[Tensor, Tensor, Tensor, Tensor],
               x: Tuple[Tensor]) -> Tuple[dict, Optional[Tensor]]:
        """OA-MIL dispatcher.

        Computes the activation gates, drives the unified
        instance-selection routine, and assembles the ``loss_oais``
        term following the ``oaie_type`` branch.
        """
        bbox_head = self.bbox_head

        epoch_open = bbox_head._epoch + 1 >= bbox_head.oais_epoch
        oamil_active = bbox_head.oamil_lambda > 0 and epoch_open
        oais_active = bbox_head.oais_flag and epoch_open
        # OA-IE only runs when loss_oais will consume the extra scores.
        # Without oamil_active there is no consumer for iter>=1 outputs.
        oaie_active = (bbox_head.oaie_flag and
                       bbox_head._epoch + 1 >= bbox_head.oaie_epoch and
                       oamil_active)

        if oaie_active:
            assert bbox_head.oaie_type in ('refine', 'random'), (
                f'unknown oaie_type: {bbox_head.oaie_type!r}; '
                'expected "refine" or "random".')
            assert bbox_head.oaie_num >= 1, (
                f'oaie_num must be >= 1 when oaie is active; '
                f'got {bbox_head.oaie_num}.')

        if not oamil_active and not oais_active:
            return {}, None

        pseudo_bbox_targets, oais_info = self._oamil_instance_selection(
            bbox_results, rois, cls_reg_targets, x,
            compute_pseudo=oais_active,
            compute_oaie=oaie_active)

        loss_oais_dict: dict = {}
        if oamil_active and oais_info is not None:
            # Per-iteration instance scores. List length is 1 without
            # OA-IE, (oaie_num + 1) with OA-IE.
            inst_scores_list = [
                self._aggregate_instance_scores(
                    s, oais_info['bags'], oais_info['pos_labels'])
                for s in oais_info['scores_list']
            ]

            loss_oais_val = self._combine_inst_scores(
                inst_scores_list,
                oaie_type=bbox_head.oaie_type,
                oaie_num=bbox_head.oaie_num,
                oaie_coef=bbox_head.oaie_coef,
            )
            loss_oais = loss_oais_val * bbox_head.oamil_lambda

            if not torch.isfinite(loss_oais):
                raise RuntimeError(
                    f'loss_oais became non-finite ({loss_oais.item()}). '
                    'Check confidence_scores and oamil_lambda.')
            loss_oais_dict['loss_oais'] = loss_oais

            if not getattr(self, '_sanity_printed', False):
                self._oamil_log_sanity(
                    oais_info, pseudo_bbox_targets, loss_oais,
                    n_iters=len(inst_scores_list))
                self._sanity_printed = True

        return loss_oais_dict, pseudo_bbox_targets

    @staticmethod
    def _combine_inst_scores(inst_scores_list: List[Tensor],
                             oaie_type: str,
                             oaie_num: int,
                             oaie_coef: float) -> Tensor:
        """Apply the OA-MIL aggregation formula across iterations.

        Single-iteration (Step 3 behavior, no OA-IE):
            loss = 1 - inst_scores_list[0]

        Multi-iteration with ``oaie_type == 'refine'``:
            loss = (1 - s_0) + (1 - mean(s_1 ... s_N)) * oaie_coef

        Multi-iteration with ``oaie_type == 'random'``:
            loss = 1 - mean(s_0 ... s_N)
        """
        if len(inst_scores_list) == 1:
            return 1.0 - inst_scores_list[0]

        assert len(inst_scores_list) == oaie_num + 1, (
            f'inst_scores_list length {len(inst_scores_list)} does not '
            f'match oaie_num+1 ({oaie_num + 1}).')

        if oaie_type == 'refine':
            head = 1.0 - inst_scores_list[0]
            tail = 1.0 - sum(inst_scores_list[1:]) / oaie_num
            return head + tail * oaie_coef
        else:  # 'random', already validated upstream
            return 1.0 - sum(inst_scores_list) / (oaie_num + 1)

    def _oamil_instance_selection(
        self,
        bbox_results: dict,
        rois: Tensor,
        cls_reg_targets: Tuple[Tensor, Tensor, Tensor, Tensor],
        x: Tuple[Tensor],
        compute_pseudo: bool = True,
        compute_oaie: bool = False,
    ) -> Tuple[Optional[Tensor], Optional[dict]]:
        """OA-IS + OA-IE pipeline.

        Stages (matching ref ``_instance_selection`` lines 213-304):

          A. Positive mask + gather positives.
          B. Decode noisy GTs to absolute coords.
          C. Bag identification by ``noisy_gt.sum(dim=1)``.
          D. Iteration loop of length ``iter_num``:
                iter_num = (oaie_num + 1) if compute_oaie else 1
             At each iter:
               - Decode current bbox_pred against the appropriate RoI
                 source (original rois at iter 0 or whenever
                 oaie_type=='random'; previous iter's predicted box
                 otherwise).
               - Forward the resulting box through bbox_head.
               - Slice softmax to the target class.
               - Store the predicted box and the score.
               - Update ``current_bbox_pred`` for next iter.
          E. If ``compute_pseudo``: per-bag best-instance pick from
             ITER 0's outputs (matches ref line 280), phi-blend with
             noisy_gt, re-encode to delta.

        Returns ``(pseudo_bbox_targets, oais_info)``:
          - ``oais_info['scores_list']`` is a list of (P,) tensors, one
            per iteration, with grad flow into the bbox_head.
          - ``oais_info['bags']`` and ``oais_info['pos_labels']`` are
            shared across iterations (bag identity is fixed once the
            noisy GTs are decoded at iter 0).
          - ``oais_info['_sample_best_score']`` carries the first bag's
            best_score for the one-shot sanity log (None if
            compute_pseudo=False).
        """
        bbox_head = self.bbox_head
        labels, _, bbox_targets, _ = cls_reg_targets
        bbox_pred = bbox_results['bbox_pred']

        assert not bbox_head.reg_class_agnostic, (
            'OA-MIL assumes reg_class_agnostic=False. Got True.')
        assert not bbox_head.reg_decoded_bbox, (
            'OA-MIL assumes reg_decoded_bbox=False (targets in delta '
            'space). Got True.')

        # A. Positive mask
        pos_mask = (labels >= 0) & (labels < bbox_head.num_classes)
        if not pos_mask.any():
            return None, None

        pos_inds = pos_mask.nonzero(as_tuple=False).squeeze(1)
        pos_labels = labels[pos_inds]
        pos_rois = rois[pos_inds]                     # (P, 5)
        pos_rois_xyxy = pos_rois[:, 1:]               # (P, 4)
        pos_bbox_targets = bbox_targets[pos_inds]     # (P, 4) delta
        pos_bbox_pred = bbox_pred.view(
            bbox_pred.size(0), -1, 4)[pos_inds, pos_labels]   # (P, 4)

        # B. Decode noisy GTs to absolute coords
        noisy_gt_boxes = bbox_head.bbox_coder.decode(
            pos_rois_xyxy, pos_bbox_targets)

        # C. Bag identification by noisy-GT coordinate sum
        _, bag_indices = torch.unique(
            noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)
        unique_bag_ids = bag_indices.unique()
        bags = [
            (bag_indices == b).nonzero(as_tuple=False).squeeze(1)
            for b in unique_bag_ids
        ]

        # D. Iteration loop
        iter_num = (bbox_head.oaie_num + 1) if compute_oaie else 1
        oaie_bboxes_list: List[Tensor] = []
        oaie_scores_list: List[Tensor] = []
        current_bbox_pred = pos_bbox_pred   # iter 0 uses original deltas
        current_roi: Optional[Tensor] = None
        pos_idx_arange = torch.arange(pos_inds.size(0),
                                      device=pos_inds.device)

        for i in range(iter_num):
            # Choose RoI source for the decode step
            if i == 0 or bbox_head.oaie_type == 'random':
                decode_source = pos_rois_xyxy
            else:
                decode_source = current_roi[:, 1:]

            new_pred_boxes = bbox_head.bbox_coder.decode(
                decode_source, current_bbox_pred).clamp(min=0.0)
            oaie_bboxes_list.append(new_pred_boxes)

            current_roi = pos_rois.clone()
            current_roi[:, 1:] = new_pred_boxes

            new_bbox_results = self._bbox_forward(x, current_roi)
            scores = torch.softmax(
                new_bbox_results['cls_score'], dim=1)[
                    pos_idx_arange, pos_labels]
            oaie_scores_list.append(scores)

            # Stage current_bbox_pred for the next iter (filtered to
            # the target class).
            current_bbox_pred = new_bbox_results['bbox_pred'].view(
                new_bbox_results['bbox_pred'].size(0), -1, 4)[
                    pos_idx_arange, pos_labels]

        assert len(oaie_bboxes_list) == iter_num
        assert len(oaie_scores_list) == iter_num

        # E. Pseudo-GT from iter 0 (matches ref line 280).
        pseudo_bbox_targets: Optional[Tensor] = None
        sample_best_score: Optional[Tensor] = None
        if compute_pseudo:
            iter0_pred_boxes = oaie_bboxes_list[0]
            iter0_scores = oaie_scores_list[0]

            pseudo_bbox_targets = torch.zeros_like(pos_bbox_targets)
            for in_bag in bags:
                bag_scores = iter0_scores[in_bag]
                bag_pred_boxes = iter0_pred_boxes[in_bag]

                best_pos = bag_scores.argmax()
                best_score = bag_scores[best_pos].detach()
                best_pred_box = bag_pred_boxes[best_pos].detach().view(1, -1)

                if sample_best_score is None:
                    sample_best_score = best_score

                phi = (best_score ** bbox_head.oais_gamma).clamp(
                    max=bbox_head.oais_theta)

                noisy_gt_this = noisy_gt_boxes[in_bag[0]].view(1, -1)
                pseudo_gt = best_pred_box * phi + noisy_gt_this * (1.0 - phi)

                pseudo_targets = bbox_head.bbox_coder.encode(
                    pos_rois_xyxy[in_bag],
                    pseudo_gt.repeat(in_bag.size(0), 1),
                )
                pseudo_bbox_targets[in_bag] = pseudo_targets

            assert pseudo_bbox_targets.shape == pos_bbox_targets.shape
            assert torch.isfinite(pseudo_bbox_targets).all(), (
                'pseudo_bbox_targets contains NaN/Inf — check oais_gamma, '
                'oais_theta and bbox_coder behavior on degenerate predicted '
                'boxes.')

        oais_info = dict(
            scores_list=oaie_scores_list,
            bags=bags,
            pos_labels=pos_labels,
            _sample_best_score=sample_best_score,
        )
        return pseudo_bbox_targets, oais_info

    @staticmethod
    def _aggregate_instance_scores(scores: Tensor,
                                   bags: List[Tensor],
                                   pos_labels: Tensor) -> Tensor:
        """MIL aggregation matching the ref ``_get_instance_cls_scores``
        (lines 306–325 of the OA-MIL paper code).

        Three nested reductions:
          1. Per-bag (instance) — max over RoIs in the bag.
          2. Per-class — mean of instance maxes per class.
          3. Across classes — mean of per-class means.

        Grad flows from ``scores`` through max → mean → mean.
        """
        inst_scores = torch.stack([scores[ib].max() for ib in bags])
        inst_labels = torch.stack([pos_labels[ib[0]] for ib in bags])

        unique_cls = inst_labels.unique()
        cls_scores_sum = inst_scores.new_zeros(())
        for c in unique_cls:
            cls_scores_sum = cls_scores_sum + \
                inst_scores[inst_labels == c].mean()
        return cls_scores_sum / unique_cls.numel()

    @staticmethod
    def _oamil_log_sanity(oais_info: dict,
                          pseudo_bbox_targets: Optional[Tensor],
                          loss_oais: Tensor,
                          n_iters: int) -> None:
        """One-shot wiring check, printed at the first iteration where
        ``oamil_active`` is True. Reports ``requires_grad`` of the four
        tensors we care about, plus the OA-IE iteration count.
        """

        def grad_str(t: Optional[Tensor]) -> str:
            if t is None:
                return 'N/A'
            return str(t.requires_grad)

        logger = MMLogger.get_current_instance()
        logger.info(
            '[OA-MIL sanity] '
            f'confidence_scores.requires_grad='
            f'{grad_str(oais_info["scores_list"][0])}, '
            f'best_score.requires_grad='
            f'{grad_str(oais_info.get("_sample_best_score"))}, '
            f'pseudo_bbox_targets.requires_grad='
            f'{grad_str(pseudo_bbox_targets)}, '
            f'loss_oais.requires_grad={grad_str(loss_oais)}, '
            f'n_oaie_iters={n_iters}')
