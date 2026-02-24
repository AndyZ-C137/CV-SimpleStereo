# Andy Zhao
"""
Stabilization: remove high-frequency jitter while keeping the intended motion.

  1) Track a camera trajectory over time (e.g., cumulative translation)
  2) Smooth that trajectory (low-pass filter)
  3) Apply a correction so actual trajectory follows the smoothed trajectory

This file implements tiny, reusable 1D smoothers for scalar signals like:
  - tx trajectory (horizontal translation)
  - ty trajectory (vertical translation)

Stabilization for Peleg is mainly *vertical-only*:
  - It removes vertical jitter (ty)
  - It avoids fighting the intended horizontal motion from rotation
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional

import numpy as np


# ---------- Moving Average Smoother ----------
@dataclass
class MovingAverage1D:
    """
    Keeps a sliding window of the most recent values and outputs their mean.
        y_t = mean(x_{t-window+1}, ..., x_t)

    If x_t is a jittery trajectory (like cumulative ty), then the moving average
    removes rapid changes and keeps the slow trend.

    Parameters:
    - window: Number of recent samples to average.
        Larger window => more smoothing, but more lag
        Smaller window => less smoothing, more responsive

    Samples stored in a deque with max_len=window.
    """
    window: int = 15

    def __post_init__(self) -> None:
        # Make sure window makes sense
        if self.window < 1:
            raise ValueError("MovingAverage1D.window must be >= 1")

        # Deque buffer holds the last `window` values.
        self._buf: Deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        """
        Reset internal state

        Use when:
          - starting a new video
          - seeking to a different time
          - reinitializing stabilization
        """
        self._buf.clear()

    def update(self, x: float) -> float:
        """
        Add a new sample and return the current smoothed value.

        1) append x
        2) return mean of buffer
        """
        self._buf.append(float(x))
        return float(np.mean(self._buf))


# Exponential Smoother
@dataclass
class ExpSmoother1D:
    """
    Exponential smoothing (a simple low-pass filter):
        s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    - alpha close to 1.0:
        s_t follows x_t closely (less smoothing)
    - alpha small (e.g. 0.1 ~ 0.3):
        s_t changes slowly (more smoothing)

    Moving average has a "hard window" and creates lag proportional to window size.
    Exponential smoothing has a "soft window" and is often smoother / simpler to tune.

    Parameters:
    - alpha:
      Must be in (0, 1].
      Typical stabilization values:
        0.15 ~ 0.35
    """
    alpha: float = 0.2

    def __post_init__(self) -> None:
        # alpha must be meaningful
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("ExpSmoother1D.alpha must be in (0, 1]")

        # Internal state: last smoothed value
        # None: "uninitialized"
        self._state: Optional[float] = None


    def reset(self) -> None:
        """
        Reset internal state.
        After reset, the next update(x) will set s = x.
        """
        self._state = None

    def update(self, x: float) -> float:
        """
        Add a new sample and return smoothed value.

        - First call: s = x (no previous state)
        - Later: apply the smoothing recurrence

        Returns: smoothed scalar float
        """
        x = float(x)

        if self._state is None:
            # First sample: initialize without lag
            self._state = x
        else:
            # self._state = self.alpha * self._state + (1 - self.alpha) * x
            self._state = self.alpha * x + (1 - self.alpha) * self._state

        return float(self._state)


