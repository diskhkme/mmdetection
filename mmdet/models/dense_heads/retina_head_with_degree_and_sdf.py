import warnings
from typing import List, Optional, Tuple, Union
import copy
import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from torch import Tensor
import numpy as np

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from ..task_modules.prior_generators import (AnchorGenerator,
                                             anchor_inside_flags)
from ..task_modules.samplers import PseudoSampler
from ..utils import images_to_levels, multi_apply, unmap
from .base_dense_head import BaseDenseHead
from .retina_head_with_degree import RetinaHeadWithDegree   
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)
from mmdet.structures import SampleList
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmcv.ops import batched_nms

@MODELS.register_module()
class RetinaHeadWithDegreeAndSDF(RetinaHeadWithDegree):
    def __init__(self, num_degree_labels, num_classes, in_channels, 
                 loss_degree_labels=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_sdf_guide=dict(type='SDFGuidedLoss', loss_weight=1.0),
                 **kwargs):
        super().__init__(num_degree_labels, num_classes, in_channels, loss_degree_labels, **kwargs)
        self.loss_sdf_guide = MODELS.build(loss_sdf_guide)

        # TODO: (KHK) sdf 정보를 통해 conv head로 x,y,w,h offset을 추정하는 head를 추가한다면?
        # 이를 구현한다면 설계상 현재 loss에서 수행하는 sdf image에 대한 roialignrotated를 별도 함수로 두고
        # 그 head와 loss에서 재사용가능하도록 구현하는 것이 좋을 듯

    def loss_by_feat(self,
                    cls_scores,
                    bbox_preds,
                    degree_labels_preds,
                    batch_gt_instances,
                    batch_img_metas,
                    batch_gt_instances_ignore=None):

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

        sdfs = [torch.stack([d['sdf'] for d in batch_img_metas])
                                  for _ in range(self.prior_generator.num_levels)]
        
        # sdfs 텐서를 device 장치로 옮기기
        sdfs = [sdf.to(device) for sdf in sdfs]
        
        # sdfs를 (batch, 1, h, w) 차원의 단일 텐서로 변환
        sdfs = torch.stack(sdfs, dim=0)  # (num_levels, batch, h, w)
        sdfs = sdfs.unsqueeze(2)  # (num_levels, batch, 1, h, w)
        
        losses_cls, losses_bbox, losses_degree_labels, loss_sdf_guide = multi_apply(
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
            sdfs,
            avg_factor=avg_factor)

        # angle loss 추가
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox,
                      loss_degree_labels=losses_degree_labels,
                      loss_sdf_guide = loss_sdf_guide)

        return losses


    def loss_by_feat_single(self,
                            cls_score: Tensor,
                            degree_label_pred: Tensor,
                            bbox_pred: Tensor,
                            anchors: Tensor,
                            labels: Tensor,
                            degree_label_targets: Tensor,
                            label_weights: Tensor,
                            degree_label_weights: Tensor,
                            bbox_targets: Tensor,
                            bbox_weights: Tensor,
                            sdfs: Tensor,
                            avg_factor: int) -> tuple:
        loss_cls, loss_bbox, loss_degree_labels = super().loss_by_feat_single(cls_score,
                                                                            degree_label_pred,
                                                                            bbox_pred,
                                                                            anchors,
                                                                            labels,
                                                                            degree_label_targets,
                                                                            label_weights,
                                                                            degree_label_weights,
                                                                            bbox_targets,
                                                                            bbox_weights,
                                                                            avg_factor=avg_factor)

        degree_label_pred = degree_label_pred.permute(0, 2, 3,
                                                      1).reshape(-1, self.degree_labels_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)

        # (KHK) Decode BBox to image coordinate(?)
        anchors = anchors.reshape(-1, anchors.size(-1))
        decoded_bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        decoded_bbox_pred = get_box_tensor(decoded_bbox_pred)

        # (KHK) Convert Degree Label into Degree Value
        degree_values = degree_label_pred.argmax(dim=1).float() * (360.0 / self.degree_labels_out_channels)

        loss_sdf_guide = self.loss_sdf_guide(sdfs, decoded_bbox_pred, degree_values)

        return loss_cls, loss_bbox, loss_degree_labels, loss_sdf_guide