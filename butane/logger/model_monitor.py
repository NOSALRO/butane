import logging
from typing import Optional, List, Dict, Union, Any

class _ModelMonitor:

    def __init__(
        self,
        logger: logging.Logger,
        *,
        increase_keys: Optional[List[str]] = None,
        decrease_keys: Optional[List[str]] = None,
        tolerance: float = 0.1,
    ):
        self.logger = logger
        self._increase_keys = increase_keys if increase_keys is not None else []
        self._decrease_keys = decrease_keys if decrease_keys is not None else []
        self._tolerance = tolerance

        # State
        self.best_step = -1
        self.best_metrics = {}

    def __call__(self, step: int, status: dict) -> bool: 
        if self.best_step < 0:
            self.best_step = step
            self.best_metrics = status
            return False

        degraded = self._check_degradation(status)

        if degraded:
            self.logger.warning(f"[Monitor] Step {step}: Metrics degraded > {self._tolerance*100:.0f}% vs Step {self.best_step}")
            return True
        else:
            self.best_step = step
            self.best_metrics = status
            self.logger.info(f"[Monitor] Step {step}: Metrics OK")
            return False

    def _check_degradation(self, current: Dict[str, float]) -> bool:
        for ik in self._increase_keys:
            if ik not in current or ik not in self.best_metrics: continue
            best, new = self.best_metrics[ik], current[ik]
            if new < best * (1 - self._tolerance):
                return True

        for dk in self._decrease_keys:
            if dk not in current or dk not in self.best_metrics: continue
            best, new = self.best_metrics[dk], current[dk]
            if new > best * (1 + self._tolerance):
                return True

        return False

    def state(self):
        return {
            'best_step': self.best_step,
            'best_metrics': self.best_metrics
        }

    def load_state(self, state):
        self.best_step = state.get('best_step', -1)
        self.best_metrics = state.get('best_metrics', {})

