# OA-MIL Port to vcnc_mil — Design Document

## Goal

Port the core ideas of OA-MIL (paper: arXiv 2207.09697, code: github.com/cxliu0/OA-MIL)
into the `vcnc_mil` repo (a fork of mmdetection 3.x with mmengine).

The original OA-MIL code targets mmdetection 2.x with a bundled mmcv. The target repo
uses mmdetection 3.x with mmengine. APIs changed significantly between these versions.

We are NOT trying to reproduce the paper's numbers. We are porting only the **algorithmic
idea** so we can combine it with our existing `VCNCKMeansConfusionAwareHook` (the gate).

The two methods are complementary:
- **Gate** (our existing hook): operates BETWEEN epochs, modifies annotations,
  targets **class-label noise**.
- **OA-MIL** (new): operates DURING forward, modifies regression target,
  targets **bounding-box noise**.

Both can be enabled simultaneously.

## Reference Files (for understanding the algorithm)

The original implementation lives in these files (provided as reference, do NOT copy
verbatim — they target the old API):

1. `oamil_ref/mmdet/models/roi_heads/standard_roi_head_oamil.py`
2. `oamil_ref/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head_oamil.py`
3. `oamil_ref/configs/_base_/models/faster_rcnn_r50_fpn_voc_oamil.py`

## Algorithm in Pseudocode

For each training step:

```
1. RPN proposes candidate boxes (standard).
2. The MaxIoUAssigner pairs proposals to noisy GTs.
3. The RandomSampler picks N positives + M negatives per image.
   → These pos+neg form "rois", with "labels", "bbox_targets" (noisy GT deltas).

4. Forward bbox_head(features for rois) → cls_score, bbox_pred.

5. OA-IS (Object-Aware Instance Selection), active if epoch+1 >= oais_epoch:
   For each unique noisy GT (bag):
     a. Collect all positive rois in this bag.
     b. Decode bbox_pred for each roi → predicted refined boxes.
     c. Compute softmax(cls_score)[:, gt_class] = "score in the correct class".
     d. Pick the roi with highest score → best_pred_box, best_score.
     e. phi = clamp(best_score^gamma, max=theta)
     f. pseudo_gt = phi * best_pred_box + (1 - phi) * noisy_gt_box
     g. Encode pseudo_gt back to delta-space → pseudo_bbox_target.

6. OA-IE (Object-Aware Instance Extension), active if epoch+1 >= oaie_epoch:
   Iterate the prediction step N=oaie_num additional times, feeding back the
   predicted boxes as new rois. Collect scores from each iteration.
   This forms a list of confidence_scores (length oaie_num + 1).

7. Compute loss_oais (MIL aggregation):
   - if oaie_type == 'refine':
       loss_oais = (1 - score_iter0) + (1 - mean(scores_iter1..N)) * oaie_coef
   - if oaie_type == 'random':
       loss_oais = 1 - mean(all_scores)
   loss_oais *= oamil_lambda

8. Loss totals:
   - loss_cls  (standard CrossEntropy from cls_score vs labels)
   - loss_bbox (regression: bbox_pred vs PSEUDO_bbox_targets, not noisy targets)
   - loss_oais (the MIL term above)
```

## Files to Create in vcnc_mil

### 1. `custom_configs/models/__init__.py` (new)

```python
from .shared2fc_bbox_head_oamil import Shared2FCBBoxHeadOAMIL, ConvFCBBoxHeadOAMIL
from .standard_roi_head_oamil import StandardRoIHeadOAMIL

__all__ = ['Shared2FCBBoxHeadOAMIL', 'ConvFCBBoxHeadOAMIL', 'StandardRoIHeadOAMIL']
```

### 2. `custom_configs/models/shared2fc_bbox_head_oamil.py` (new)

This should be a SUBCLASS of mmdet 3.x's `Shared2FCBBoxHead`. It should NOT
duplicate the entire `ConvFCBBoxHead` body (the reference file does this because
mmdet 2.x was different). Inherit and only override what's needed.

What changes vs the standard `Shared2FCBBoxHead`:

- `__init__`: accept extra OA-MIL kwargs (`oamil_lambda`, `oais_flag`, `oais_epoch`,
  `oais_gamma`, `oais_theta`, `oaie_flag`, `oaie_num`, `oaie_coef`, `oaie_epoch`,
  `oaie_type`). Store them as attributes. Pass the rest to `super().__init__`.

- `loss_and_target` (in mmdet 3.x this is the method that combines target generation
  and loss computation, replacing the old separate `get_targets` + `loss`): accept
  an optional `pseudo_bbox_targets` parameter. If provided, use it instead of the
  computed `bbox_targets` for `loss_bbox`. Same logic as in reference file's `loss()`
  function lines 270-301.

- IMPORTANT: in mmdet 3.x, the bbox_head's `loss()` method receives a `SamplingResultList`
  and returns a `dict[str, Tensor]`. The signature is something like:
  ```
  def loss(self, cls_score, bbox_pred, rois, sampling_results, rcnn_train_cfg, ...)
  ```
  Check `mmdet/models/roi_heads/bbox_heads/bbox_head.py` in the target repo (vcnc_mil)
  for the exact current signature before writing this.

- Initialize `self._epoch = 0` as an attribute (will be updated by a hook).

### 3. `custom_configs/models/standard_roi_head_oamil.py` (new)

Subclass of mmdet 3.x's `StandardRoIHead`. Override `loss()` method (NOT `forward_train`,
which is the mmdet 2.x name).

The crux of the work is in three methods:

#### `loss(self, x, rpn_results_list, batch_data_samples)`
(this is the mmdet 3.x replacement for `forward_train`)

```
Steps inside loss():
  1. assign + sample (use self.bbox_assigner + self.bbox_sampler — same as standard)
  2. _bbox_loss(x, sampling_results)  ← this is where OA-MIL kicks in
  3. Return losses dict
```

In mmdet 3.x, the standard `_bbox_loss` does:
1. Get rois from sampling_results
2. Forward bbox_head → cls_score, bbox_pred
3. Call `bbox_head.loss_and_target(cls_score, bbox_pred, rois, sampling_results, ...)`

You need to insert the OA-MIL logic BETWEEN steps 2 and 3:
- Run `_oamil_instance_selection` to compute `pseudo_bbox_targets` and `loss_oais`
- Pass `pseudo_bbox_targets` to `bbox_head.loss_and_target`
- Add `loss_oais` to the returned losses dict

#### `_oamil_instance_selection(self, x, rois, cls_score, bbox_pred, bbox_targets, labels)`

Implement following lines 213-304 of the reference file, but using mmdet 3.x APIs.
Key points:

- Bags are identified by `torch.unique(noisy_gt_boxes.sum(dim=1))` — this is a hack to
  group rois that come from the same GT box. Keep this exact logic.
- For each bag, do the OA-IS computation: phi = clamp(score^gamma, max=theta),
  pseudo_gt = phi * best_pred + (1-phi) * noisy_gt.
- Encode the pseudo_gt back to delta-space using `self.bbox_head.bbox_coder.encode`.
- Return `pseudo_bbox_targets` (same shape as the positive part of `bbox_targets`).

#### `_oamil_instance_extension(self, x, rois, labels, pos_inds, current_pred_boxes)`

Implement following lines 246-268. This is the optional iteration step (OA-IE).
- Iterate `oaie_num` times.
- Each iteration: build new rois from current_pred_boxes, forward through bbox_head,
  compute new confidence scores.
- Returns a list of (predicted_boxes, confidence_scores) for each iteration.

#### `_compute_loss_oais(self, oaie_scores_list, pos_labels, uniq_inst, pos_indices)`

Lines 195-207 in reference. Computes the MIL aggregation loss.

### 4. `custom_configs/hooks/oamil_epoch_injector_hook.py` (new)

A small hook to inject the current epoch into `bbox_head._epoch`. Necessary because
the OA-MIL ROI head uses `self._epoch` to gate OA-IS/OA-IE activation.

```python
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class OAMILEpochInjectorHook(Hook):
    """Inject current epoch number into bbox_head for OA-MIL gating."""
    def before_train_epoch(self, runner):
        bbox_head = runner.model.roi_head.bbox_head
        bbox_head._epoch = runner.epoch  # 0-indexed
```

### 5. `custom_configs/exp_configs/exp_oamil_voc_sim40.py` (new)

A config file for the experiment. Should:
- Import `model = dict(...)` with `roi_head.type='StandardRoIHeadOAMIL'` and
  `bbox_head.type='Shared2FCBBoxHeadOAMIL'`.
- Set the OA-MIL hyperparameters in `bbox_head` matching the reference VOC config
  (oamil_lambda=0.1, oais_epoch=2, oaie_epoch=9, oaie_num=4, etc.).
- Include `OAMILEpochInjectorHook` in `custom_hooks`.
- Optionally include the existing `VCNCKMeansConfusionAwareHook` to test the combo.
- Add `custom_imports` pointing to the new modules.

## Implementation Order

Do this in steps, testing each. Do NOT write everything at once.

### Step 1: Skeleton only
- Create the 3 files with class definitions, `__init__` storing OA-MIL params, but
  override no methods yet. The forward should be identical to the parent class.
- Register with `@MODELS.register_module()`.
- Create the config pointing to these classes with `oamil_lambda=0.0` (effectively disabled).
- Run a training step. Verify it produces the SAME loss as the standard model.
  This validates that the registration and config wiring work.

### Step 2: OA-IS only
- Add the instance selection logic (no OA-IE iteration, no OA-IS loss yet).
- Just compute the pseudo_bbox_targets and pass to bbox_head.loss.
- Set `oais_epoch=2`, `oaie_flag=False`, `oamil_lambda=0` (no extra loss).
- Run a few epochs. Verify the model still trains (loss decreasing, mAP > 0).
- The behavior should differ from baseline starting at epoch 3 (when OA-IS activates).

### Step 3: Loss OA-IS (without OA-IE)
- Add the `loss_oais` computation, single-iteration version.
- Set `oamil_lambda=0.1`, `oaie_flag=False`.
- Train and verify `loss_oais` appears in the log, decreasing over time.

### Step 4: Full OA-IE
- Add the iteration logic with `oaie_num=4`.
- Set `oaie_flag=True`, `oaie_epoch=9`, `oaie_type='refine'`.
- Verify the iteration kicks in at epoch 10.

### Step 5: Combine with gate
- Add `VCNCKMeansConfusionAwareHook` (with the FIXED config: E1=False, E4=False,
  aggressive_only=True) alongside the OAMILEpochInjectorHook.
- Verify both hooks fire and the training runs end-to-end.

## Critical Things to Watch

1. **Don't copy `ConvFCBBoxHeadOAMIL` verbatim.** The reference file reimplements the
   entire ConvFC head because in mmdet 2.x the base class had different internals.
   In mmdet 3.x, inherit from `Shared2FCBBoxHead` directly. The only override you
   need is in `loss` (or `loss_and_target`, depending on the API version).

2. **bbox_coder.decode signatures.** In mmdet 3.x, `bbox_coder.decode(rois, bbox_pred)`
   may have a slightly different argument order. Check `mmdet/models/task_modules/coders/`
   in vcnc_mil.

3. **Sampling results structure.** In mmdet 3.x, `SamplingResult` has `pos_priors`
   instead of `pos_bboxes` in some versions, and `pos_gt_bboxes` was renamed in others.
   Check the actual structure by inspecting one in a debugger or print statement.

4. **The `_epoch` attribute.** Make sure `OAMILEpochInjectorHook` is registered in
   `custom_imports` and added to `custom_hooks` in the config. Without it, the head
   never activates OA-IS/OA-IE.

5. **First epoch behavior.** With `oais_epoch=2`, the OA-MIL logic only activates at
   epoch 2 (0-indexed, so epoch index 1 in some interpretations). Verify which
   convention is used. The reference code does `epoch+1 >= oais_epoch`, suggesting
   1-indexed comparison.

## Validation Plan

After Step 1: train 1 epoch with `oamil_lambda=0`, compare loss values with baseline
Faster R-CNN. Should match within numerical precision.

After Step 3: train 12 epochs on sym40. Compare:
- Baseline Faster R-CNN: 0.725 (from the planilha)
- VCNC c30 (existing gate, no OA-MIL): 0.78 (from the planilha)
- OA-MIL alone: ???  (unknown — this is novel territory since OA-MIL was designed
  for bbox noise, not class noise)
- OA-MIL + gate: ???  (the goal of the integration)

If OA-MIL alone gives ≤ baseline in sym40, that's expected — OA-MIL is not designed
for class noise. The interesting test is on a dataset with BOTH bbox noise AND class
noise, which the Filipe's dataset has.

## Notes Specific to This Project

- The Filipe's dataset has multiple noise types (loc, missing, bkg, sym, asym). OA-MIL
  is specifically targeted at LOC noise. So we expect the largest gain in scenarios
  where loc noise is present.
- The existing gate handles class noise. So the combination is expected to dominate
  in scenarios with BOTH types of noise.
