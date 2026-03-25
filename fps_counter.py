import time
from collections import deque


class FPSCounter:
    def __init__(self, window=60):
        self._timestamps = deque(maxlen=window)
        self._fps = 0.0

    def tick(self):
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > 1:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        return self._fps
