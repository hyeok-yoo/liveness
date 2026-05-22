"""POS rPPG 알고리즘 (Wang et al. 2017)."""
from __future__ import annotations

import numpy as np


def pos_pulse(rgb: np.ndarray, fs: float = 30.0, window_sec: float = 1.6) -> np.ndarray:
    """RGB 시계열 → pulse 신호 (POS 알고리즘).

    Parameters
    ----------
    rgb : np.ndarray
        Shape (T, 3), float, [R, G, B] 순서.
    fs : float
        샘플링 주파수 (fps).
    window_sec : float
        슬라이딩 윈도우 길이 (초).

    Returns
    -------
    np.ndarray
        Shape (T,) pulse 신호.
    """
    rgb = np.nan_to_num(rgb, nan=0.0).astype(np.float64)  # NaN 안전 처리
    T = rgb.shape[0]
    window = max(2, int(window_sec * fs))

    H = np.zeros(T, dtype=np.float64)

    # POS 투영 행렬 (논문 Table 1)
    P = np.array([[0.0, 1.0, -1.0],
                  [-2.0, 1.0, 1.0]], dtype=np.float64)

    for t in range(0, T - window + 1):
        C = rgb[t: t + window]  # (window, 3)
        mean_C = C.mean(axis=0)
        # 채널 평균이 0이면 분할 불가 → 스킵
        if np.any(mean_C == 0):
            continue
        Cn = C / mean_C  # (window, 3) 정규화

        S = P @ Cn.T  # (2, window)
        std0 = S[0].std()
        std1 = S[1].std()
        alpha = (std0 / std1) if std1 > 1e-9 else 1.0

        h = S[0] + alpha * S[1]
        h -= h.mean()

        # overlap-add
        H[t: t + window] += h

    return H


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    T, fs = 300, 30.0
    t = np.arange(T) / fs
    # 심박 60 BPM = 1 Hz 사인파 + 노이즈
    r = 100 + 5 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 1, T)
    g = 80 + 4 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 1, T)
    b = 60 + 3 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 1, T)
    rgb = np.stack([r, g, b], axis=1)
    pulse = pos_pulse(rgb, fs=fs)
    assert pulse.shape == (T,), f"unexpected shape {pulse.shape}"
    print("pos self-test passed. pulse std:", pulse.std())
