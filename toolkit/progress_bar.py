from tqdm import tqdm
import time


def is_progress_update_due(step, start_step, total_steps, every):
    """Return whether a completed training step should refresh progress."""
    return (
        step == start_step
        or (step + 1) % every == 0
        or step + 1 >= total_steps
    )


def materialize_metrics(metrics):
    """Convert detached scalar tensors only when telemetry consumes them."""
    if metrics is None:
        return None
    return metrics.__class__(
        (key, value.item() if hasattr(value, 'item') else value)
        for key, value in metrics.items()
    )


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._time()

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            self.start_t += cur_t - self.last_time
            self.last_print_t = cur_t

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)
