"""CHROM rPPG 알고리즘 (de Haan & Jeanne 2013)."""
from __future__ import annotations

import numpy as np

from .signal_processing import bandpass


def chrom_pulse(rgb: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """RGB 시계열 → pulse 신호 (CHROM 알고리즘).

    Parameters
    ----------
    rgb : np.ndarray
        Shape (T, 3), float, [R, G, B] 순서.
    fs : float
        샘플링 주파수 (fps).

    Returns
    -------
    np.ndarray
        Shape (T,) pulse 신호.
    """
    rgb = np.nan_to_num(rgb, nan=0.0).astype(np.float64)

    mean_C = rgb.mean(axis=0)
    # 채널 평균이 0이면 정규화 불가 → 1로 대체
    mean_C[mean_C == 0] = 1.0
    Cn = rgb / mean_C  # (T, 3)

    R, G, B = Cn[:, 0], Cn[:, 1], Cn[:, 2]

    Xs = 3 * R - 2 * G
    Ys = 1.5 * R + G - 1.5 * B

    # 심박 주파수 대역으로 제한 후 alpha 계산 (잡음 억제)
    Xs_f = bandpass(Xs, low_hz=0.7, high_hz=4.0, fs=fs)
    Ys_f = bandpass(Ys, low_hz=0.7, high_hz=4.0, fs=fs)

    std_y = Ys_f.std()
    alpha = Xs_f.std() / std_y if std_y > 1e-9 else 1.0

    return Xs_f - alpha * Ys_f


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    T, fs = 300, 30.0
    t = np.arange(T) / fs
    r = 100 + 5 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 1, T)
    g = 80 + 4 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 1, T)
    b = 60 + 3 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 1, T)
    rgb = np.stack([r, g, b], axis=1)
    pulse = chrom_pulse(rgb, fs=fs)
    assert pulse.shape == (T,), f"unexpected shape {pulse.shape}"
    print("chrom self-test passed. pulse std:", pulse.std())
