import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from ..task_modules.prior_generators import (AnchorGenerator,
                                             anchor_inside_flags)
from ..task_modules.samplers import PseudoSampler
from ..utils import images_to_levels, multi_apply, unmap
from .base_dense_head import BaseDenseHead
from .retina_head import RetinaHead
from mmcv.cnn import ConvModule



@MODELS.register_module()
class RetinaHeadWithDegree(RetinaHead):
    def __init__(self, num_degree_labels, num_classes, in_channels, 
                 loss_degree_labels=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        # init_cfg 설정을 먼저 정의
        if 'init_cfg' not in kwargs:
            kwargs['init_cfg'] = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=[
                    dict(
                        type='Normal',
                        name='retina_cls',
                        std=0.01,
                        bias_prob=0.01),
                    dict(
                        type='Normal',
                        name='retina_degree_labels',
                        std=0.01,
                        bias_prob=0.01)
                ])
        elif 'override' not in kwargs['init_cfg']:
            kwargs['init_cfg']['override'] = [
                dict(
                    type='Normal',
                    name='retina_degree_labels',
                    std=0.01,
                    bias_prob=0.01)
            ]
        else:
            kwargs['init_cfg']['override'].append(
                dict(
                    type='Normal',
                    name='retina_degree_labels',
                    std=0.01,
                    bias_prob=0.01)
            )
            
        super().__init__(num_classes, in_channels, **kwargs)
        self.num_degree_labels = num_degree_labels
        # angle 분류 출력 채널 수 (배경포함이면 angle_bins+1)
        if self.use_sigmoid_cls:
            self.degree_labels_out_channels = num_degree_labels
        else:
            self.degree_labels_out_channels = num_degree_labels + 1
        # angle loss 초기화
        self.loss_degree_labels = MODELS.build(loss_degree_labels)
        # 각도 분류 서브넷 레이어 초기화
        self._init_degree_labels_layers()

    def _init_degree_labels_layers(self):
        self.degree_labels_convs = nn.ModuleList()
        in_channels = self.in_channels
        for i in range(self.stacked_convs):  # 예: stacked_convs=4 활용 또는 별도 설정
            self.degree_labels_convs.append(
                ConvModule(in_channels, self.feat_channels, 3, stride=1, padding=1, 
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            in_channels = self.feat_channels
        self.retina_degree_labels = nn.Conv2d(in_channels, self.num_base_priors * self.degree_labels_out_channels, 3, padding=1)

    def forward_single(self, x):
        # 기존 RetinaHead의 cls/reg 처리
        cls_score, bbox_pred = super().forward_single(x)  # (또는 cls_feat/reg_feat 구분하여 처리)
        # Angle 분류 branch 처리
        degree_feat = x
        for conv in self.degree_labels_convs:
            degree_feat = conv(degree_feat)
        degree_labels_prec = self.retina_degree_labels(degree_feat)
        return cls_score, bbox_pred, degree_labels_prec
    
    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        # TODO: (KHK) 백그라운드로 채우는 것이 맞을지? 0으로 채우는 것이 맞을지?
        degree_labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_degree_labels,
                                  dtype=torch.long)
        degree_label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)


        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            degree_labels[pos_inds] = sampling_result.pos_gt_degree_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
                degree_label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
                degree_label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
            degree_label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            degree_labels = unmap(degree_labels, num_total_anchors, inside_flags)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            degree_label_weights = unmap(degree_label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, degree_labels, degree_label_weights, pos_inds,
                neg_inds, sampling_result)
    
    def get_targets(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True,
                    return_sampling_results: bool = False) -> tuple:
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - avg_factor (int): Average factor that is used to average
                  the loss. When using sampling method, avg_factor is usually
                  the sum of positive and negative priors. When using
                  `PseudoSampler`, `avg_factor` is usually equal to the number
                  of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         all_degree_labels, all_degree_label_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:9]
        rest_results = list(results[9:])  # user-added return values
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # update `_raw_positive_infos`, which will be used when calling
        # `get_positive_infos`.
        self._raw_positive_infos.update(sampling_results=sampling_results_list)
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        degree_labels_targets_list = images_to_levels(all_degree_labels,
                                             num_level_anchors)
        degree_label_weights_list = images_to_levels(all_degree_label_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, degree_labels_targets_list, degree_label_weights_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)
    
    def loss_by_feat(self,
                    cls_scores,
                    bbox_preds,
                    degree_labels_preds,
                    batch_gt_instances,
                    batch_img_metas,
                    batch_gt_instances_ignore=None):
        """Loss function.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            angle_preds (list[Tensor]): Angle predictions for each scale
                level with shape (N, num_anchors * angle_bins, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, 
         degree_labels_targets_list, degree_label_weights_list,
         avg_factor) = cls_reg_targets
        
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        
        # # 기본 loss 계산
        # losses_cls, losses_bbox = multi_apply(
        #     self.loss_by_feat_single,
        #     cls_scores,
        #     bbox_preds,
        #     all_anchor_list,
        #     labels_list,
        #     label_weights_list,
        #     bbox_targets_list,
        #     bbox_weights_list,
        #     avg_factor=avg_factor)
        #
        # # # angle loss 계산 - labels_list와 같은 구조로 처리
        # losses_degree_labels = multi_apply(
        #     self.loss_by_feat_single_degree_labels,
        #     degree_labels_preds,
        #     degree_labels_targets_list,
        #     degree_label_weights_list,
        #     avg_factor=avg_factor)

        # OK
        # for i in range(len(degree_labels_preds)):
        #     self.loss_by_feat_single_degree_labels(degree_labels_preds[i], degree_labels_targets_list[i], degree_label_weights_list[i], avg_factor)

        losses_cls, losses_bbox, losses_degree_labels = multi_apply(
                self.loss_by_feat_single,
                cls_scores,
                degree_labels_preds,
                bbox_preds,
                all_anchor_list,
                labels_list,
            degree_labels_targets_list,
                label_weights_list,
            degree_label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                avg_factor=avg_factor)


        # angle loss 추가
        # losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

        # # angle loss 추가
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_degree_labels=losses_degree_labels)
        # print(losses_degree_labels)
        return losses


    def loss_by_feat_single(self,
                            cls_score: Tensor, degree_label_pred: Tensor,
                            bbox_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            degree_label_targets: Tensor,
                            label_weights: Tensor,
                            degree_label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        loss_cls, loss_bbox = super().loss_by_feat_single(cls_score,
                                            bbox_pred,
                                            anchors,
                                            labels,
                                            label_weights,
                                            bbox_targets,
                                            bbox_weights,
                                            avg_factor=avg_factor)

        degree_label_targets = degree_label_targets.reshape(-1)
        degree_label_weights = degree_label_weights.reshape(-1)
        degree_label_pred = degree_label_pred.permute(0, 2, 3, 1).reshape(-1, self.degree_labels_out_channels)

        # loss 계산
        loss_degree_labels = self.loss_degree_labels(
            degree_label_pred, degree_label_targets, degree_label_weights, avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_degree_labels

    def loss_by_feat_single_degree_labels(self, degree_labels_pred: Tensor, 
                                degree_labels_targets: Tensor,
                                degree_label_weights: Tensor, avg_factor: int) -> tuple:
        """Calculate the angle loss of a single scale level.

        Args:
            angle_pred (Tensor): Angle predictions for each scale level
                with shape (N, num_anchors * angle_bins, H, W).
            degree_labels_targets (Tensor): Angle targets for each scale level
                with shape (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            Tensor: Angle loss.
        """

        degree_labels_targets = degree_labels_targets.reshape(-1)
        degree_label_weights = degree_label_weights.reshape(-1)
        degree_labels_pred = degree_labels_pred.permute(0, 2, 3, 1).reshape(-1, self.degree_labels_out_channels)

        # loss 계산
        loss_degree_labels = self.loss_degree_labels(
            degree_labels_pred, degree_labels_targets, degree_label_weights, avg_factor=avg_factor)
        
        return loss_degree_labels