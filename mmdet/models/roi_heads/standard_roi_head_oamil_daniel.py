# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead


@MODELS.register_module()
class StandardRoIHeadOAMILDANIEL(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        # mmdetection/mmdet/engine/hooks/set_epoch_info_hook.py
        #self.bbox_head.current_epoch  = 0 
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_mask_head(self, mask_roi_extractor: ConfigType,
                       mask_head: ConfigType) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict or ConfigDict): Config of mask roi
                extractor.
            mask_head (dict or ConfigDict): Config of mask in mask head.
        """
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = MODELS.build(mask_head)

    # TODO: Need to refactor later
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results, rois, bbox_targets = self.bbox_loss(x, sampling_results)
            #losses.update(bbox_results['loss_bbox'])

            # apply OA-MIL
            loss_oamil, pseudo_bbox_targets = self._oamil(bbox_targets, bbox_results, rois, x)

            # compute classification and localization loss
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets, pseudo_bbox_targets=pseudo_bbox_targets)

            losses.update(loss_bbox)
            losses.update(loss_oamil)

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor], sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        bbox_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)

        return bbox_results, rois, bbox_targets

    def _oamil(self, bbox_targets, bbox_results, rois, x):
        """
        Procedure:
            1. perform instance selection
            2. compute instance selection loss
        """
        loss_bbox = dict()

        # lambda controls whether to perform OA-MIL
        if self.bbox_head.oamil_lambda > 0:
            '''
            1. Perform instance selection
            '''
            # get bbox targets
            labels, cur_bbox_targets = bbox_targets[0], bbox_targets[2]
            pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
            pos_labels = labels[pos_inds.type(torch.bool)]

            # get indices of unique gt boxes
            pos_bbox_targets = cur_bbox_targets[pos_inds.type(torch.bool)]
            noisy_gt_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_targets)
            uniq_inst, pos_indices = torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)
            
            # perform Object-Aware Instance Selection (OA-IS)
            oaie_scores_list, pseudo_bbox_targets = self._instance_selection(bbox_results, labels, rois, x, cur_bbox_targets)

            '''
            2. Compute instance selection loss
            '''
            inst_scores_list = []
            for confidence_scores in oaie_scores_list:
                inst_scores = self._get_instance_cls_scores(pos_labels, confidence_scores, uniq_inst, pos_indices)
                inst_scores_list.append(inst_scores)
            
            if len(inst_scores_list) > 1:
                if self.bbox_head.oaie_type == 'refine':
                    loss_bbox['loss_oais'] = (1-inst_scores_list[0]) + (1-sum(inst_scores_list[1:])/self.bbox_head.oaie_num)*self.bbox_head.oaie_coef
                elif self.bbox_head.oaie_type == 'random':
                    loss_bbox['loss_oais'] = 1 - sum(inst_scores_list)/(self.bbox_head.oaie_num+1)
            elif len(inst_scores_list) == 1:
                loss_bbox['loss_oais'] = 1-inst_scores_list[0]
            else:
                loss_bbox['loss_oais'] = torch.tensor([1.0], device="cuda")
            loss_bbox['loss_oais'] *= self.bbox_head.oamil_lambda
        else:
            pseudo_bbox_targets = None

        return loss_bbox, pseudo_bbox_targets

    def _instance_selection(self, bbox_results, labels, rois, x, cur_bbox_targets):
        """
        Procedure of instance selection:
            1. construct object bags 
            2. apply instance selector (OA-IE is optional in step 2)
            3. get best selected instances using Eq. (4)
        """

        '''
        1. Construct object bags
        '''
        # get indices of object bags from noisy gt
        pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
        inds = torch.ones(pos_inds.sum()).cuda()
        pos_bbox_targets = cur_bbox_targets[pos_inds.type(torch.bool)]
        noisy_gt_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_targets)
        uniq_inst, pos_indices = torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)
        object_bag_indices = [torch.where(pos_indices == inst)[0] for inst in torch.unique(pos_indices)]

        # initialize bbox results for object bags
        new_bbox_results = {}
        new_bbox_results['cls_score'], new_bbox_results['bbox_pred'] = bbox_results['cls_score'], bbox_results['bbox_pred']
        
        # keep positive instances
        new_bbox_results['bbox_pred'] = new_bbox_results['bbox_pred'].view(new_bbox_results['bbox_pred'].size(0), -1, 4)[pos_inds.type(torch.bool)]
        
        '''
        2. Apply instance selector
            - OA-IE is optional in this step
        '''
        # The number of iteration depends on whether to perform Object-Aware Instance Extension (OA-IE)
        #   - iter num=1 if w/o OA-IE 
        #   - iter num=N+1 if with OA-IE, where N is the number of OA-IE
        oaie_bboxes_list, oaie_scores_list = [], []
        iter_num = self.bbox_head.oaie_num+1 if self.bbox_head.oaie_flag and self.bbox_head.current_epoch +1 >= self.bbox_head.oaie_epoch else 1
        for i in range(iter_num):
            # get prediction of each instance
            bbox_pred = new_bbox_results['bbox_pred']
            # print(bbox_pred.size())
            #empty_tensor = False
            if bbox_pred.nelement() == 0:
                continue
            #    empty_tensor = True
            #    bbox_pred = torch.zeros(1, 20, 4, device="cuda")
            inds = torch.ones(bbox_pred.size(0)).type(torch.bool).cuda()
            pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[inds, labels[pos_inds.type(torch.bool)]]

            # decode prediction of each instance
            if i == 0 or self.bbox_head.oaie_type == 'random':
                new_pred_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_pred)
            else:
                new_pred_boxes = self.bbox_head.bbox_coder.decode(new_roi[:, 1:], pos_bbox_pred)

            new_roi = rois[pos_inds.type(torch.bool)].clone()
            new_roi[:,1:] = new_pred_boxes
            oaie_bboxes_list.append(new_pred_boxes)

            #if empty_tensor:
            #    confidence_scores = torch.zeros(1, device="cuda")
            #    oaie_scores_list.append(confidence_scores)
            #    continue

            # apply instance selector and output confidence scores
            # NOTE: instance selector shares the same parameters with classifier
            new_bbox_results = self._bbox_forward(x, new_roi)
            confidence_scores = torch.softmax(new_bbox_results['cls_score'], dim=1)[inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
            oaie_scores_list.append(confidence_scores)

            #print(new_pred_boxes)
            #print(confidence_scores.size())
            #print(confidence_scores, "\n\n\n")

        '''
        3. Get best selected instances
        '''
        # perform OA-IS
        oais_flag = self.bbox_head.oais_flag and (self.bbox_head.current_epoch +1 >= self.bbox_head.oais_epoch)
        if oais_flag and len(oaie_bboxes_list) > 0 and len(oaie_scores_list) > 0:
            # to save best selected instances
            pseudo_bbox_targets = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda()
            pseudo_gt_boxes = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda()

            new_pred_boxes, confidence_scores = oaie_bboxes_list[0], oaie_scores_list[0]
            # iterate over each object bag
            for index in object_bag_indices:
                # get confidence score of each instance in an object bag
                object_bag_boxes = new_pred_boxes[index]
                object_bag_scores = confidence_scores[index]

                # get best instance in each object bag
                _, max_inds = torch.max(object_bag_scores, dim=0)
                best_score = object_bag_scores[max_inds].clone()
                best_instance = object_bag_boxes[max_inds].clone()
                best_instance = best_instance.clamp(min=0.0).view(1, -1)

                # modify noisy gt using Eq. (4)
                phi = ((best_score.detach())**self.bbox_head.oais_gamma).clamp(max=self.bbox_head.oais_theta)
                best_selected_instance = best_instance.detach() * phi + noisy_gt_boxes[index[0]].view(1, -1) * (1 - phi)

                # reset grount-truth according to best selected instance
                pseudo_gt_targets = self.bbox_head.bbox_coder.encode(rois[:, 1:][pos_inds.type(torch.bool)][index], best_selected_instance.repeat(len(index), 1))
                pseudo_bbox_targets[index] = pseudo_gt_targets
                pseudo_gt_boxes[index] = best_selected_instance.repeat(len(index), 1)
        else:
            pseudo_bbox_targets = None

        return oaie_scores_list, pseudo_bbox_targets

    def _get_instance_cls_scores(self, pos_labels, confidence_scores, uniq_inst, pos_indices):
        """Compute confidence scores of object bags in training."""
        # instance-level confidence scores
        inst_labels = []
        inst_scores = []
        for inst in torch.unique(pos_indices):
            inst_inds = torch.where(pos_indices == inst)[0]
            inst_scores.append(confidence_scores[inst_inds].max().view(-1))
            inst_labels.append(pos_labels[inst_inds[0]].view(-1))

        inst_labels = torch.cat(inst_labels, dim=0)
        inst_scores = torch.cat(inst_scores, dim=0)

        # class-level confidence scores
        cls_scores = 0
        for cls in torch.unique(inst_labels):
            cls_inds = torch.where(inst_labels == cls)[0]
            cls_scores += inst_scores[cls_inds].mean()
        cls_scores /= len(torch.unique(inst_labels))
        return cls_scores



    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats)
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats)
        return mask_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list

    def predict_bbox_logits(self,
                        x,
                        batch_img_metas,
                        rpn_results_list,
                        rcnn_test_cfg,
                        rescale=False):

        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return [], []

        bbox_results = self._bbox_forward(x, rois)

        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']

        num_proposals_per_img = tuple(len(p) for p in proposals)

        rois_split = rois.split(num_proposals_per_img, 0)
        cls_scores_split = cls_scores.split(num_proposals_per_img, 0)

        if bbox_preds is not None:
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds_split = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds_split = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds_split = (None,) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois_split,
            cls_scores=cls_scores_split,
            bbox_preds=bbox_preds_split,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)

    return result_list, cls_scores_split

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list
