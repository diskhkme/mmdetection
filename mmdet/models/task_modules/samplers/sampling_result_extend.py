import warnings

import numpy as np
import torch
from torch import Tensor

from .sampling_result import SamplingResult
from ..assigners import AssignResult, AssignResultExtend

class SamplingResultExtend(SamplingResult):
    def __init__(self,
                 pos_inds: Tensor,
                 neg_inds: Tensor,
                 priors: Tensor,
                 gt_bboxes: Tensor,
                 assign_result: AssignResultExtend,
                 gt_flags: Tensor,
                 avg_factor_with_neg: bool = True) -> None:
        super().__init__(pos_inds, neg_inds, priors, gt_bboxes, assign_result, gt_flags, avg_factor_with_neg)
        self.pos_gt_degree_labels = assign_result.degree_labels[pos_inds]

    # @property
    # def angles(self):
    #     """torch.Tensor: concatenated positive and negative priors"""
    #     return torch.cat([self.pos_angles, self.neg_angles])

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_priors': self.pos_priors,
            'neg_priors': self.neg_priors,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
            'num_pos': self.num_pos,
            'num_neg': self.num_neg,
            'avg_factor': self.avg_factor
        }
    