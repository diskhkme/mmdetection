import logging
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

# 중첩된 속성에 안전하게 접근하기 위한 헬퍼 함수 (선택적)
import functools
import operator

from mmdet.registry import HOOKS

def get_nested_attr(obj, attr_path):
    """'.'으로 구분된 경로 문자열을 사용하여 중첩된 속성을 가져옵니다."""
    try:
        return functools.reduce(getattr, attr_path.split('.'), obj)
    except AttributeError:
        return None

def set_nested_attr(obj, attr_path, value):
    """'.'으로 구분된 경로 문자열을 사용하여 중첩된 속성을 설정합니다."""
    try:
        attrs = attr_path.split('.')
        target_obj = functools.reduce(getattr, attrs[:-1], obj)
        setattr(target_obj, attrs[-1], value)
        return True
    except AttributeError:
        return False

@HOOKS.register_module()
class LossWeightSchedulerHook(Hook):
    def __init__(self,
                 # 예: 'roi_head.bbox_head.loss_cls.loss_weight' 또는 'bbox_head.loss_bbox.loss_weight'
                 loss_attr_path,
                 # 예: [(epoch1, weight1), (epoch2, weight2), ...]
                 weight_schedule,
                 by_epoch=True,
                 verbose=False, # 로그 출력 여부
                 use_linear_interpolation=False, # 선형 보간 사용 여부
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_attr_path = loss_attr_path
        # 스케줄은 (단계, 값)의 리스트이며, 단계 오름차순으로 정렬되어야 함
        self.weight_schedule = sorted(weight_schedule, key=lambda x: x[0])
        self.by_epoch = by_epoch
        self.verbose = verbose
        self.use_linear_interpolation = use_linear_interpolation

    def _get_scheduled_weight(self, runner):
        if self.by_epoch:
            current_step = runner.epoch
        else:
            current_step = runner.iter

        # 첫 번째 스케줄 이전이면 첫 번째 값 사용
        if current_step < self.weight_schedule[0][0]:
            return self.weight_schedule[0][1]
        
        # 마지막 스케줄 이후면 마지막 값 사용
        if current_step >= self.weight_schedule[-1][0]:
            return self.weight_schedule[-1][1]
        
        # 선형 보간 사용 여부에 따라 처리
        if self.use_linear_interpolation:
            # 현재 단계가 속한 구간 찾기
            for i in range(len(self.weight_schedule) - 1):
                step1, weight1 = self.weight_schedule[i]
                step2, weight2 = self.weight_schedule[i + 1]
                
                if step1 <= current_step < step2:
                    # 선형 보간 계산
                    t = (current_step - step1) / (step2 - step1)
                    interpolated_weight = weight1 + t * (weight2 - weight1)
                    return interpolated_weight
        
        # 선형 보간을 사용하지 않거나 구간을 찾지 못한 경우
        # 현재 단계보다 큰 첫 번째 스케줄의 이전 값을 사용
        for i, (step, weight) in enumerate(self.weight_schedule):
            if current_step < step:
                if i > 0:
                    return self.weight_schedule[i-1][1]
                else:
                    return self.weight_schedule[0][1]
        
        # 모든 경우를 처리했으므로 여기까지 오면 안 됨
        return self.weight_schedule[-1][1]

    def _apply_schedule(self, runner):
        target_weight = self._get_scheduled_weight(runner)

        # runner.model은 보통 DataParallel 또는 DistributedDataParallel로 감싸져 있음
        model_module = runner.model.module if hasattr(runner.model, 'module') else runner.model

        # 중첩 속성 설정 시도
        success = set_nested_attr(model_module, self.loss_attr_path, target_weight)

        step_type = "epoch" if self.by_epoch else "iter"
        current_step = runner.epoch if self.by_epoch else runner.iter
        
        if self.verbose:
            if success:
                logging.info(f"Loss weight at {step_type} {current_step}: {target_weight:.4f}")
            else:
                logging.warning(f"Failed to set loss weight at {step_type} {current_step}")

    # 에폭 시작 또는 이터레이션 시작 시 가중치 조절
    def before_train_epoch(self, runner):
        if self.by_epoch:
            self._apply_schedule(runner)
            
    def before_train_iter(self, runner, batch_idx: int,
                          data_batch = None):
        if not self.by_epoch:
            self._apply_schedule(runner)

