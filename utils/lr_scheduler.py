"""Learning-rate schedules used by GenDental."""

import math

from torch.optim.lr_scheduler import LRScheduler


class HorizonAwareWarmupCosineLR(LRScheduler):
    """Linear warmup followed by a non-restarting cosine decay.

    Schedule values come from the current run, not a restored checkpoint.
    After max_epochs the LR remains at eta_min instead of rising again.
    """

    _SCHEDULE_KEYS = (
        "warmup_epochs",
        "max_epochs",
        "warmup_start_lr",
        "eta_min",
    )

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=0.0,
        eta_min=0.0,
        last_epoch=-1,
    ):
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative.")
        if self.max_epochs <= self.warmup_epochs:
            raise ValueError(
                "max_epochs must be greater than warmup_epochs."
            )
        super().__init__(optimizer, last_epoch)

    def _lr_at_epoch(self, base_lr):
        epoch = max(int(self.last_epoch), 0)
        if self.warmup_epochs and epoch < self.warmup_epochs:
            ratio = epoch / float(self.warmup_epochs)
            return self.warmup_start_lr + ratio * (
                base_lr - self.warmup_start_lr
            )

        decay_epochs = self.max_epochs - self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / float(decay_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.eta_min + (base_lr - self.eta_min) * cosine

    def get_lr(self):
        return [self._lr_at_epoch(base_lr) for base_lr in self.base_lrs]

    def load_state_dict(self, state_dict):
        # Restore progress, but never let an old checkpoint override the
        # training horizon explicitly configured for the current run.
        schedule = {
            key: getattr(self, key) for key in self._SCHEDULE_KEYS
        }
        restored = dict(state_dict)
        for key in self._SCHEDULE_KEYS:
            restored.pop(key, None)
        super().load_state_dict(restored)
        for key, value in schedule.items():
            setattr(self, key, value)

        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._last_lr = lrs
