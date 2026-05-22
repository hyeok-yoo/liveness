"""rPPG 공통 신호 처리 유틸리티.

10초 캡처 같은 짧은 신호에서도 안정적인 BPM/SNR을 내도록 강화:
  - smooth-priors detrending (Tarvainen 2002)
  - Hanning window + zero-padding 후 FFT (spectral leakage 억제, 분해능 향상)
  - harmonic-aware SNR (기본파 + 2nd harmonic 합을 신호로 인정)
  - parabolic peak interpolation (이산 FFT bin보다 정밀한 BPM)
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ----------------------------------------------------------------------
# Filters & detrending
# ----------------------------------------------------------------------


def bandpass(
    sig: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass 필터 (zero-phase, filtfilt)."""
    nyq = fs / 2.0
    lo = np.clip(low_hz / nyq, 1e-4, 1 - 1e-4)
    hi = np.clip(high_hz / nyq, 1e-4, 1 - 1e-4)
    if lo >= hi:
        raise ValueError(f"bandpass: low_hz({low_hz}) must be < high_hz({high_hz})")
    b, a = sp_signal.butter(order, [lo, hi], btype="band")
    padlen = min(3 * max(len(a), len(b)), len(sig) - 1)
    if padlen < 1:
        return sig.copy()
    return sp_signal.filtfilt(b, a, sig, padlen=padlen)


def detrend(sig: np.ndarray, lambda_: float = 100.0) -> np.ndarray:
    """Smooth-priors detrending (Tarvainen 2002).

    저주파 트렌드를 강하게 제거해서 베이스라인 드리프트로 인한
    저주파 누설을 줄인다. 짧은 캡처(10초)일 때 효과 큼.

    Parameters
    ----------
    sig : np.ndarray  Shape (T,)
    lambda_ : float
        regularization (클수록 강한 detrending). 100 권장.
    """
    T = sig.shape[0]
    if T < 4:
        return sp_signal.detrend(sig, type="linear")

    I = sparse.eye(T, format="csc")
    # second-order difference matrix D2 (T-2, T)
    e = np.ones(T)
    D2 = sparse.diags([e, -2 * e, e], offsets=[0, 1, 2], shape=(T - 2, T), format="csc")
    A = I + (lambda_ ** 2) * (D2.T @ D2)
    trend = spsolve(A, sig)
    return np.asarray(sig - trend, dtype=np.float64)


# ----------------------------------------------------------------------
# Spectrum (Hanning + zero-padding) — 한 번 만들어 BPM/SNR가 공유
# ----------------------------------------------------------------------


def _power_spectrum(
    sig: np.ndarray, fs: float, pad_factor: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Hanning window + zero-padding 후 power spectrum.

    Returns
    -------
    (freqs, power) : tuple[np.ndarray, np.ndarray]
        실수 FFT 주파수 축, power.
    """
    N = len(sig)
    if N < 2:
        return np.zeros(1), np.zeros(1)
    x = sig - sig.mean()
    win = np.hanning(N)
    xw = x * win
    nfft = max(N * pad_factor, 1024)
    spec = np.fft.rfft(xw, n=nfft)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, power


def _parabolic_interp(power: np.ndarray, idx: int) -> float:
    """3-point parabolic interpolation → sub-bin offset (단위: bin)."""
    if idx <= 0 or idx >= len(power) - 1:
        return 0.0
    y0, y1, y2 = power[idx - 1], power[idx], power[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


# ----------------------------------------------------------------------
# BPM & SNR
# ----------------------------------------------------------------------


def compute_bpm(
    sig: np.ndarray,
    fs: float,
    low_bpm: float = 42.0,
    high_bpm: float = 180.0,
) -> tuple[float, float]:
    """Hanning + zero-padding FFT + 포물선 보간 기반 BPM 추정.

    Returns
    -------
    (bpm, peak_freq_hz)
    """
    freqs, power = _power_spectrum(sig, fs)
    if len(freqs) < 2:
        return 0.0, 0.0

    low_hz = low_bpm / 60.0
    high_hz = high_bpm / 60.0
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return 0.0, 0.0

    band_idx = np.where(mask)[0]
    local_peak = int(np.argmax(power[band_idx]))
    peak_idx = int(band_idx[local_peak])

    # parabolic interpolation
    df = freqs[1] - freqs[0]
    offset = _parabolic_interp(power, peak_idx)
    peak_freq = float(freqs[peak_idx] + offset * df)
    bpm = peak_freq * 60.0
    return float(bpm), peak_freq


def compute_snr_db(
    sig: np.ndarray,
    fs: float,
    peak_bpm: float,
    band_bpm_width: float = 12.0,
    valid_range: tuple[float, float] = (42.0, 180.0),
    include_harmonic: bool = True,
) -> float:
    """Harmonic-aware SNR (dB).

    fundamental ± band/2 + (옵션) 2nd harmonic ± band/2 합을 signal,
    valid_range 내 나머지를 noise로 정의.

    실제 PPG는 1차+2차 harmonic에 에너지가 분산되므로 2차까지 신호로 인정.
    """
    freqs, power = _power_spectrum(sig, fs)
    if len(freqs) < 2 or peak_bpm <= 0:
        return -999.0

    low_valid = valid_range[0] / 60.0
    high_valid = valid_range[1] / 60.0
    valid_mask = (freqs >= low_valid) & (freqs <= high_valid)

    half_hz = band_bpm_width / 2.0 / 60.0
    peak_hz = peak_bpm / 60.0
    sig_mask = (freqs >= peak_hz - half_hz) & (freqs <= peak_hz + half_hz)

    if include_harmonic:
        h2 = 2.0 * peak_hz
        if h2 <= high_valid:
            sig_mask |= (freqs >= h2 - half_hz) & (freqs <= h2 + half_hz)

    sig_mask &= valid_mask
    noise_mask = valid_mask & ~sig_mask

    s = float(power[sig_mask].sum())
    n = float(power[noise_mask].sum())
    if n < 1e-12:
        return float("inf")
    if s < 1e-12:
        return -999.0
    return float(10.0 * np.log10(s / n))


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs = 30.0
    T = 300  # 10s
    t = np.arange(T) / fs
    sig = np.sin(2 * np.pi * 1.2 * t) + 0.3 * rng.normal(size=T)

    det = detrend(sig)
    filt = bandpass(det, 0.7, 4.0, fs)
    bpm, freq = compute_bpm(filt, fs)
    snr = compute_snr_db(filt, fs, bpm)
    print(f"signal_processing self-test: BPM={bpm:.2f} (expected ~72), "
          f"freq={freq:.3f} Hz, SNR={snr:.1f} dB")
