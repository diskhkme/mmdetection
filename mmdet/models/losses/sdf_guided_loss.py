from typing import Optional, Tuple

import torch
torch.cuda.empty_cache()  # CUDA 메모리 캐시 비우기
torch.cuda.memory_summary()
import torch.nn as nn
from torch import Tensor
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from mmdet.registry import MODELS
from .utils import weighted_loss
from mmcv.ops.roi_align_rotated import RoIAlignRotated


@weighted_loss
def sdf_guided_loss(sdf_img: Tensor,
                    decoded_bbox_pred: Tensor,
                    degree_values: Tensor = None,
                    roi_align_rotated=None) -> Tensor:
    """SDF Guided Loss function.

    Args:
        sdf_img (Tensor): The SDF image tensor
        decoded_bbox_pred (Tensor): The predicted bounding boxes
        degree_values (Tensor, optional): Rotation angles in degrees
        roi_align_rotated: RoIAlignRotated instance

    Returns:
        Tensor: Calculated loss
    """
    # sdf_img를 -1에서 1 사이로 정규화
    sdf_min = sdf_img.min()
    sdf_max = sdf_img.max()
    if sdf_max > sdf_min:  # 0으로 나누기 방지
        sdf_img = -1.0 + 2.0 * (sdf_img - sdf_min) / (sdf_max - sdf_min)

    # sdf_img를 decoded_bbox_pred와 같은 데이터 타입(float32)으로 변환
    if sdf_img.dtype != decoded_bbox_pred.dtype:
        sdf_img = sdf_img.to(dtype=decoded_bbox_pred.dtype)

    # decoded_bbox_pred는 minx, miny, maxx, maxy 형식으로 되어 있음
    minx = decoded_bbox_pred[:, 0]
    miny = decoded_bbox_pred[:, 1]
    maxx = decoded_bbox_pred[:, 2]
    maxy = decoded_bbox_pred[:, 3]

    # 중심점 계산
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    # 너비와 높이 계산
    w = maxx - minx
    h = maxy - miny

    # RoI 형식으로 변환 (batch_index, cx, cy, w, h, angle)
    rois = decoded_bbox_pred.new_zeros(decoded_bbox_pred.shape[0], 6)
    
    # batch 크기 계산
    batch_size = sdf_img.shape[0]
    # 각 배치당 RoI 개수 계산
    rois_per_batch = decoded_bbox_pred.shape[0] // batch_size
    
    # 각 RoI에 대해 올바른 배치 인덱스 할당
    for batch_idx in range(batch_size):
        start_idx = batch_idx * rois_per_batch
        end_idx = (batch_idx + 1) * rois_per_batch
        rois[start_idx:end_idx, 0] = batch_idx
        
    rois[:, 1] = cx
    rois[:, 2] = cy
    rois[:, 3] = w
    rois[:, 4] = h

    # degree_values가 None이면 0도로 설정
    if degree_values is None:
        rois[:, 5] = 0.0
    else:
        # degree를 radian으로 변환 (1도 = π/180 라디안)
        rois[:, 5] = degree_values * (torch.pi / 180.0)

    margin = 10.0
    # 새로운 rois 텐서 생성
    new_rois = rois.clone()
    
    # width와 height는 양수여야 함
    new_rois[:, 3:5] = torch.clamp(rois[:, 3:5], min=margin, max=799.0 - margin)
    
    # 박스가 이미지 경계를 벗어나지 않도록 중심점 클램프
    half_w = new_rois[:, 3] / 2.0
    half_h = new_rois[:, 4] / 2.0
    
    # cx 클램프: 박스의 좌우 경계가 이미지 안에 있도록
    new_rois[:, 1] = torch.clamp(
        rois[:, 1],
        min=half_w + margin,
        max=799.0 - half_w - margin
    )
    # cy 클램프: 박스의 상하 경계가 이미지 안에 있도록
    new_rois[:, 2] = torch.clamp(
        rois[:, 2],
        min=half_h + margin,
        max=799.0 - half_h - margin
    )

    # sdf_feature 추출
    try:
        sdf_feature = roi_align_rotated(sdf_img, new_rois)
        
        # NaN 체크 및 디버깅
        if torch.isnan(sdf_feature).any():
            # (n,1,10,10) -> (n,) 형태로 변환하여 NaN 체크
            nan_indices = torch.where(torch.isnan(sdf_feature).reshape(sdf_feature.size(0), -1).any(dim=1))[0]
            print(f"\n=== NaN 발생 디버깅 정보 ===")
            print(f"NaN이 발생한 RoI 개수: {len(nan_indices)}")
            for idx in nan_indices:
                print(f"\nRoI {idx} 정보:")
                print(f"원본 RoI 좌표: {rois[idx]}")
                print(f"클램핑된 RoI 좌표: {new_rois[idx]}")
                print(f"SDF 특징값 통계:")
                print(f"- 최소값: {sdf_feature[idx].min().item()}")
                print(f"- 최대값: {sdf_feature[idx].max().item()}")
                print(f"- 평균값: {sdf_feature[idx].mean().item()}")
                print(f"- NaN 개수: {torch.isnan(sdf_feature[idx]).sum().item()}")
    except RuntimeError as e:
        print(f"RoIAlignRotated 오류: {e}")
        print(f"sdf_img shape: {sdf_img.shape}, dtype: {sdf_img.dtype}")
        print(f"rois shape: {rois.shape}, dtype: {rois.dtype}")
        print(f"rois 값 범위: min={rois.min().item()}, max={rois.max().item()}")
        raise

    # RoI별 loss 계산
    loss = sdf_feature.view(sdf_feature.size(0), -1).mean(dim=1)
    
    # NaN 체크 및 디버깅
    if torch.isnan(loss).any():
        nan_indices = torch.where(torch.isnan(loss))[0]
        print(f"\n=== 손실값 NaN 발생 디버깅 정보 ===")
        print(f"NaN이 발생한 손실값 개수: {len(nan_indices)}")
        for idx in nan_indices:
            print(f"\nRoI {idx} 손실값 정보:")
            print(f"RoI 좌표: {new_rois[idx]}")
            print(f"손실값: {loss[idx].item()}")
            print(f"SDF 특징값 통계:")
            print(f"- 최소값: {sdf_feature[idx].min().item()}")
            print(f"- 최대값: {sdf_feature[idx].max().item()}")
            print(f"- 평균값: {sdf_feature[idx].mean().item()}")

    return loss


@MODELS.register_module()
class SDFGuidedLoss(nn.Module):
    """SDF Guided Loss module.

    Args:
        method (str): Loss calculation method. Options are 'full' and 'boundary'.
        output_size (Tuple[int, int]): Output size of RoIAlign operation.
        reduction (str): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float): The weight of loss.
    """

    def __init__(self,
                 method: str = 'full',
                 output_size: Tuple[int, int] = (10, 10),
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.method = method
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.roi_align_output_size = output_size
        self.roi_align_rotated = RoIAlignRotated(
            output_size=output_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )

    def forward(self,
                sdf_img: Tensor,
                decoded_bbox_pred: Tensor,
                degree_values: Tensor = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            sdf_img (Tensor): The SDF image tensor
            decoded_bbox_pred (Tensor): The predicted bounding boxes
            degree_values (Tensor, optional): Rotation angles in degrees
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        sdf_guide_loss = sdf_guided_loss(
            sdf_img,
            decoded_bbox_pred,
            degree_values,
            roi_align_rotated=self.roi_align_rotated,
            reduction=reduction,
            avg_factor=avg_factor)


        range = (self.roi_align_output_size[0] * self.roi_align_output_size[1])
        sdf_guide_loss_area_normalized = (sdf_guide_loss + range) / (2*range)

        return self.loss_weight * sdf_guide_loss_area_normalized