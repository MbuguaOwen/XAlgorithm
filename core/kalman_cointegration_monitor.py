
import numpy as np
from collections import deque
from core.adaptive_filters import KalmanFilter

class KalmanCointegrationMonitor:
    def __init__(self, window_size=100, threshold=0.1):
        self.window_size = window_size
        self.kalman = KalmanFilter()
        self.spread_window = deque(maxlen=window_size)
        self.threshold = threshold

    def update(self, spread: float) -> float:
        """
        Updates the Kalman filter and returns a normalized cointegration stability score ∈ [0, 1].
        A high score means the spread is likely stationary (mean-reverting).
        """
        filtered_spread = self.kalman.update(spread)
        self.spread_window.append(filtered_spread)

        if len(self.spread_window) < self.window_size:
            return 0.0

        mean = np.mean(self.spread_window)
        std = np.std(self.spread_window)
        z_scores = [(x - mean) / std if std > 1e-6 else 0.0 for x in self.spread_window]

        # Stability: how many z-scores stay within ±threshold?
        stable_count = sum(abs(z) < self.threshold for z in z_scores)
        stability_score = stable_count / self.window_size

        return round(stability_score, 4)
