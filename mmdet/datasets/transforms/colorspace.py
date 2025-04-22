# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional
import cv2
from torch import Tensor

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS
from .augment_wrappers import _MAX_LEVEL, level_to_mag

import scipy.ndimage
import torch


def _pad_to_size(tensor, target_img_size):
    """패딩을 적용하여 텐서를 지정된 크기로 만듭니다."""
    h, w = tensor.shape[:2]
    target_h, target_w = target_img_size
    if h >= target_h and w >= target_w:
        return tensor
        
    # 경계 픽셀 값 추출
    bottom_edge = tensor[-1, :]  # 하단 경계
    right_edge = tensor[:, -1]  # 우측 경계
    
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    # 하단 패딩
    if pad_h > 0:
        bottom_pad = bottom_edge.unsqueeze(0).repeat(pad_h, 1)
        tensor = torch.cat([tensor, bottom_pad], dim=0)
        
    # 우측 패딩
    if pad_w > 0:
        right_pad = right_edge.unsqueeze(1).repeat(1, pad_w)
        tensor = torch.cat([tensor, right_pad], dim=1)
        
    return tensor


def _generate_sdf(input_image_ndarray: np.ndarray, bw_threshold: int = 100) -> Tensor:
    """이미지에서 SDF(Signed Distance Function)를 생성합니다."""
    # --- 입력 유효성 검사 ---
    if not isinstance(input_image_ndarray, np.ndarray):
        print("Error: Input must be a NumPy ndarray.")
        return
    if input_image_ndarray.ndim < 2 or input_image_ndarray.ndim > 3:
        print(f"Error: Input ndarray has unsupported dimensions: {input_image_ndarray.ndim}")
        return
    
    image = input_image_ndarray # 입력 ndarray를 사용
    
    # --- 그레이스케일 변환 ---
    if image.ndim == 3:
        # 3차원 배열이면 컬러 이미지로 간주하고 그레이스케일로 변환
        # OpenCV는 기본적으로 BGR 순서를 가정함
        try:
            # 채널 수가 3 또는 4가 아니면 에러 발생 가능성 있음
            if image.shape[2] == 4: # RGBA 경우, 알파 채널 제외
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 3: # BGR 또는 RGB
                # 만약 입력이 RGB 순서라면 cv2.COLOR_RGB2GRAY 사용
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            else:
                print(f"Error: Input image has unsupported channel count: {image.shape[2]}")
                return
        except cv2.error as e:
            print(f"Error during grayscale conversion: {e}")
            return
            
    elif image.ndim == 2:
        # 2차원 배열이면 이미 그레이스케일로 간주
        gray_image = image
    else: 
        # 위에서 이미 ndim 체크했지만, 명확성을 위해 추가
        print(f"Error: Unexpected image dimensions: {image.ndim}")
        return

    # --- 1단계: 이진 마스크 생성 ---
    # 예시: 간단한 임계값 처리 (값이 100보다 어두운 픽셀을 '검은색'으로 간주)
    threshold_value = bw_threshold
    binary_mask = gray_image < threshold_value

    # --- 2단계: 거리 변환 및 SDF 계산 ---
    dist_inside = scipy.ndimage.distance_transform_edt(binary_mask)
    dist_outside = scipy.ndimage.distance_transform_edt(~binary_mask)
    sdf = dist_outside - dist_inside

    # 내부는 -1로 truncate하고 외부만 0~1로 정규화
    sdf_outside = dist_outside.copy()
    sdf_outside_max = sdf_outside.max()
    if sdf_outside_max > 0:  # 0으로 나누기 방지
        sdf_outside = sdf_outside / sdf_outside_max  # 0~1로 정규화

    # 최종 SDF: 내부는 -1, 외부는 0~1
    normalized_sdf = np.where(binary_mask, -1.0, sdf_outside)

    return to_tensor(normalized_sdf)


@TRANSFORMS.register_module()
class GenerateSDF(BaseTransform):
    """Generate SDF for the image."""
    def __init__(self,
                 bw_threshold: int = 100,
                 target_img_size: tuple = (800, 800),
                 pad_to_size: bool = True
                 ) -> None:
        self.bw_threshold = bw_threshold
        self.pad_to_size = pad_to_size
        self.target_img_size = target_img_size

    def transform(self, results: dict) -> dict:
        img = results['img']
        sdf = _generate_sdf(img, self.bw_threshold)
        
        if self.pad_to_size and sdf.shape != self.target_img_size:
            sdf = _pad_to_size(sdf, self.target_img_size)
            
        results['sdf'] = sdf
        return results


@TRANSFORMS.register_module()
class ColorTransform(BaseTransform):
    """Base class for color transformations. All color transformations need to
    inherit from this base class. ``ColorTransform`` unifies the class
    attributes and class functions of color transformations (Color, Brightness,
    Contrast, Sharpness, Solarize, SolarizeAdd, Equalize, AutoContrast, Invert,
    and Posterize), and only distort color channels, without impacting the
    locations of the instances.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing the geometric
            transformation and should be in range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for color transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for color transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0 <= prob <= 1.0, f'The probability of the transformation ' \
                                 f'should be in range [0,1], got {prob}.'
        assert level is None or isinstance(level, int), \
            f'The level should be None or type int, got {type(level)}.'
        assert level is None or 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range [0,{_MAX_LEVEL}], got {level}.'
        assert isinstance(min_mag, float), \
            f'min_mag should be type float, got {type(min_mag)}.'
        assert isinstance(max_mag, float), \
            f'max_mag should be type float, got {type(max_mag)}.'
        assert min_mag <= max_mag, \
            f'min_mag should smaller than max_mag, ' \
            f'got min_mag={min_mag} and max_mag={max_mag}'
        self.prob = prob
        self.level = level
        self.min_mag = min_mag
        self.max_mag = max_mag

    def _transform_img(self, results: dict, mag: float) -> None:
        """Transform the image."""
        pass

    @cache_randomness
    def _random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.prob

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        return level_to_mag(self.level, self.min_mag, self.max_mag)

    def transform(self, results: dict) -> dict:
        """Transform function for images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """

        if self._random_disable():
            return results
        mag = self._get_mag()
        self._transform_img(results, mag)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'level={self.level}, '
        repr_str += f'min_mag={self.min_mag}, '
        repr_str += f'max_mag={self.max_mag})'
        return repr_str


@TRANSFORMS.register_module()
class Color(ColorTransform):
    """Adjust the color balance of the image, in a manner similar to the
    controls on a colour TV set. A magnitude=0 gives a black & white image,
    whereas magnitude=1 gives the original image. The bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Color transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Color transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Color transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Color should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Color should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Apply Color transformation to image."""
        # NOTE defaultly the image should be BGR format
        img = results['img']
        results['img'] = mmcv.adjust_color(img, mag).astype(img.dtype)


@TRANSFORMS.register_module()
class Brightness(ColorTransform):
    """Adjust the brightness of the image. A magnitude=0 gives a black image,
    whereas magnitude=1 gives the original image. The bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Brightness transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Brightness transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Brightness transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Brightness should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Brightness should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Adjust the brightness of image."""
        img = results['img']
        results['img'] = mmcv.adjust_brightness(img, mag).astype(img.dtype)


@TRANSFORMS.register_module()
class Contrast(ColorTransform):
    """Control the contrast of the image. A magnitude=0 gives a gray image,
    whereas magnitude=1 gives the original imageThe bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Contrast transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Contrast transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Contrast transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Contrast should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Contrast should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Adjust the image contrast."""
        img = results['img']
        results['img'] = mmcv.adjust_contrast(img, mag).astype(img.dtype)


@TRANSFORMS.register_module()
class Sharpness(ColorTransform):
    """Adjust images sharpness. A positive magnitude would enhance the
    sharpness and a negative magnitude would make the image blurry. A
    magnitude=0 gives the origin img.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Sharpness transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Sharpness transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Sharpness transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Sharpness should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Sharpness should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Adjust the image sharpness."""
        img = results['img']
        results['img'] = mmcv.adjust_sharpness(img, mag).astype(img.dtype)


@TRANSFORMS.register_module()
class Solarize(ColorTransform):
    """Solarize images (Invert all pixels above a threshold value of
    magnitude.).

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Solarize transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Solarize transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for Solarize transformation.
            Defaults to 256.0.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 256.0) -> None:
        assert 0. <= min_mag <= 256.0, f'min_mag for Solarize should be ' \
                                       f'in range [0, 256], got {min_mag}.'
        assert 0. <= max_mag <= 256.0, f'max_mag for Solarize should be ' \
                                       f'in range [0, 256], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Invert all pixel values above magnitude."""
        img = results['img']
        results['img'] = mmcv.solarize(img, mag).astype(img.dtype)


@TRANSFORMS.register_module()
class SolarizeAdd(ColorTransform):
    """SolarizeAdd images. For each pixel in the image that is less than 128,
    add an additional amount to it decided by the magnitude.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing SolarizeAdd
            transformation. Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for SolarizeAdd transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for SolarizeAdd transformation.
            Defaults to 110.0.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 110.0) -> None:
        assert 0. <= min_mag <= 110.0, f'min_mag for SolarizeAdd should be ' \
                                       f'in range [0, 110], got {min_mag}.'
        assert 0. <= max_mag <= 110.0, f'max_mag for SolarizeAdd should be ' \
                                       f'in range [0, 110], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, results: dict, mag: float) -> None:
        """SolarizeAdd the image."""
        img = results['img']
        img_solarized = np.where(img < 128, np.minimum(img + mag, 255), img)
        results['img'] = img_solarized.astype(img.dtype)


@TRANSFORMS.register_module()
class Posterize(ColorTransform):
    """Posterize images (reduce the number of bits for each color channel).

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Posterize
            transformation. Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Posterize transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for Posterize transformation.
            Defaults to 4.0.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 4.0) -> None:
        assert 0. <= min_mag <= 8.0, f'min_mag for Posterize should be ' \
                                     f'in range [0, 8], got {min_mag}.'
        assert 0. <= max_mag <= 8.0, f'max_mag for Posterize should be ' \
                                     f'in range [0, 8], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Posterize the image."""
        img = results['img']
        results['img'] = mmcv.posterize(img, math.ceil(mag)).astype(img.dtype)


@TRANSFORMS.register_module()
class Equalize(ColorTransform):
    """Equalize the image histogram. The bboxes, masks and segmentations are
    not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Equalize transformation.
            Defaults to 1.0.
        level (int, optional): No use for Equalize transformation.
            Defaults to None.
        min_mag (float): No use for Equalize transformation. Defaults to 0.1.
        max_mag (float): No use for Equalize transformation. Defaults to 1.9.
    """

    def _transform_img(self, results: dict, mag: float) -> None:
        """Equalizes the histogram of one image."""
        img = results['img']
        results['img'] = mmcv.imequalize(img).astype(img.dtype)


@TRANSFORMS.register_module()
class AutoContrast(ColorTransform):
    """Auto adjust image contrast.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing AutoContrast should
             be in range [0, 1]. Defaults to 1.0.
        level (int, optional): No use for AutoContrast transformation.
            Defaults to None.
        min_mag (float): No use for AutoContrast transformation.
            Defaults to 0.1.
        max_mag (float): No use for AutoContrast transformation.
            Defaults to 1.9.
    """

    def _transform_img(self, results: dict, mag: float) -> None:
        """Auto adjust image contrast."""
        img = results['img']
        results['img'] = mmcv.auto_contrast(img).astype(img.dtype)


@TRANSFORMS.register_module()
class Invert(ColorTransform):
    """Invert images.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing invert therefore should
             be in range [0, 1]. Defaults to 1.0.
        level (int, optional): No use for Invert transformation.
            Defaults to None.
        min_mag (float): No use for Invert transformation. Defaults to 0.1.
        max_mag (float): No use for Invert transformation. Defaults to 1.9.
    """

    def _transform_img(self, results: dict, mag: float) -> None:
        """Invert the image."""
        img = results['img']
        results['img'] = mmcv.iminvert(img).astype(img.dtype)
