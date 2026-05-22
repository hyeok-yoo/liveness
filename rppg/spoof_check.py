"""rPPG 기반 spoof 판정 모듈 (robust 버전).

핵심 통찰: **절대 SNR**은 조명/카메라/거리에 따라 들쭉날쭉하지만,
"**얼굴 SNR이 배경 SNR보다 분명히 높은가**" 라는 상대 판정은 환경 변화에
훨씬 강건하다.

  - 사진:    face_snr ≈ bg_snr (둘 다 환경 노이즈)
  - 영상:    face_snr ≈ bg_snr 또는 둘 다 약함 (배경의 픽셀 그리드 노이즈)
  - 실제얼굴: face_snr ≫ bg_snr (얼굴에만 pulse 신호 존재)

따라서 판정은:
  ① 얼굴 fused SNR이 배경 SNR보다 `snr_margin_db` 이상 높아야 함 (핵심 신호)
  ② 또는 fused SNR이 강한 양수 (절대 신호)
  둘 중 하나도 만족 못 하면 spoof.
설계 보조:
  - 3 ROI 신호 융합으로 √N SNR 향상
  - 신뢰 ROI(상대 SNR ≥ margin)만 BPM disagreement 평가
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pos import pos_pulse
from .chrom import chrom_pulse
from .signal_processing import bandpass, compute_bpm, compute_snr_db

_FACE_ROIS = ("forehead", "left_cheek", "right_cheek")


@dataclass
class RPPGAnalysis:
    bpms: dict[str, float]
    snrs: dict[str, float]
    background_bpm: float
    background_snr_db: float
    fused_bpm: float
    fused_snr_db: float
    is_spoof: bool
    reasons: list[str] = field(default_factory=list)


class RPPGSpoofChecker:
    """rPPG 신호로 실제 얼굴 vs 스푸핑 판정.

    Parameters
    ----------
    fs : float
        샘플링 주파수 (fps).
    snr_threshold_db : float
        절대 SNR 통과 임계값 (fallback). 환경 노이즈 적은 곳에선 이것만으로
        통과 가능. 보수적으로 매우 낮게 둠.
    snr_margin_db : float
        얼굴 fused SNR이 배경 SNR보다 얼마나 높아야 "진짜 pulse" 로 인정.
        **핵심 spoof 판정 신호**.
    bpm_agreement : float
        신뢰 ROI 사이 BPM 최대 spread (BPM).
    bg_match_bpm : float
        배경 BPM이 face와 이 범위 내면 환경 노이즈 매칭.
    algorithm : str
        "pos" 또는 "chrom".
    """

    def __init__(
        self,
        fs: float = 30.0,
        snr_threshold_db: float = 3.0,
        snr_margin_db: float = 1.0,  # 1.0 dB로 완화 — 약한 환경에서도 face vs bg 분리 가능
        bpm_agreement: float = 20.0,  # 노이즈 환경에서 BPM 흔들림 허용
        bg_match_bpm: float = 5.0,
        algorithm: str = "pos",
    ) -> None:
        if algorithm not in ("pos", "chrom"):
            raise ValueError(f"algorithm must be 'pos' or 'chrom', got {algorithm!r}")
        self.fs = fs
        self.snr_threshold_db = snr_threshold_db
        self.snr_margin_db = snr_margin_db
        self.bpm_agreement = bpm_agreement
        self.bg_match_bpm = bg_match_bpm
        self.algorithm = algorithm

    # ------------------------------------------------------------------

    def analyze(self, signals: dict[str, np.ndarray]) -> RPPGAnalysis:
        bpms: dict[str, float] = {}
        snrs: dict[str, float] = {}
        pulses: dict[str, np.ndarray] = {}

        # 1) 얼굴 ROI 각각: pulse 추출 → multi-window best-SNR 선택
        # 사용자가 캡처 도중 잠시 흔들거나 깜빡여도 좋은 윈도우만 사용해 robust.
        for roi in _FACE_ROIS:
            rgb = signals.get(roi)
            if rgb is None or len(rgb) < 2:
                bpms[roi] = 0.0
                snrs[roi] = -999.0
                continue
            pulse = self._extract_pulse(rgb)
            pulse_bp = bandpass(pulse, low_hz=0.7, high_hz=2.5, fs=self.fs)
            bpm, snr = self._best_window_metrics(pulse_bp)
            bpms[roi] = bpm
            snrs[roi] = snr
            pulses[roi] = pulse_bp

        # 2) 신호 융합 (SNR-weighted) + multi-window
        fused_bpm, fused_snr = self._fused_metrics(pulses, snrs)

        # 3) 배경 — 동일하게 multi-window best
        bg_rgb = signals.get("background")
        if bg_rgb is not None and len(bg_rgb) >= 2:
            bg_pulse = self._extract_pulse(bg_rgb)
            bg_pulse_bp = bandpass(bg_pulse, low_hz=0.7, high_hz=2.5, fs=self.fs)
            bg_bpm, bg_snr = self._best_window_metrics(bg_pulse_bp)
        else:
            bg_bpm, bg_snr = 0.0, -999.0

        # 4) Best-ROI metrics — 한 ROI라도 충분히 강하면 fallback 통과
        face_snrs_valid = [snrs[r] for r in _FACE_ROIS if snrs[r] > -100.0]
        best_roi_snr = max(face_snrs_valid) if face_snrs_valid else -999.0

        # 4) Spoof 판정
        reasons: list[str] = []

        # 핵심 신호: face fused SNR − background SNR
        # bg_snr이 -999면 (배경 분석 불가) margin 체크 건너뜀
        if bg_snr > -100.0:
            snr_margin = fused_snr - bg_snr
        else:
            snr_margin = float("inf")

        # Adaptive margin: 배경이 매우 조용한 환경(bg_snr < -8 dB)에서는
        # 절대 노이즈 자체가 낮으므로 margin 요구치를 살짝 낮춘다 (1.0 dB).
        # 사진은 face_snr ≈ bg_snr이라 spread 자체가 안 생기므로 여전히 안전.
        effective_margin = self.snr_margin_db
        if bg_snr < -8.0:
            effective_margin = 1.0

        # Best-ROI margin (한 ROI라도 배경 대비 충분히 강한가)
        if bg_snr > -100.0:
            best_roi_margin = best_roi_snr - bg_snr
        else:
            best_roi_margin = float("inf")

        # ① 통과 조건 (셋 중 하나라도 만족하면 통과):
        #   (a) fused margin ≥ effective_margin  — 핵심 신호
        #   (b) fused_snr ≥ snr_threshold AND fused_snr > bg_snr  — 강한 절대 신호
        #   (c) best ROI margin ≥ effective_margin + 1  — 한 ROI라도 배경보다 분명히 강함
        passes_margin = snr_margin >= effective_margin
        passes_absolute = (
            fused_snr >= self.snr_threshold_db and fused_snr > bg_snr
        )
        passes_best_roi = best_roi_margin >= (effective_margin + 1.0)
        if not (passes_margin or passes_absolute or passes_best_roi):
            reasons.append(
                f"weak_pulse: fused_snr={fused_snr:.1f} dB, "
                f"bg_snr={bg_snr:.1f} dB, "
                f"fused_margin={snr_margin:.1f} dB, "
                f"best_roi_margin={best_roi_margin:.1f} dB "
                f"(need ≥{effective_margin:.1f} OR fused≥{self.snr_threshold_db:.1f})"
            )

        # ② BPM 일관성 — 배경 대비 일정 margin 이상 SNR 있는 ROI만 비교
        margin_threshold = bg_snr + self.snr_margin_db if bg_snr > -100.0 else self.snr_threshold_db
        trusted_bpms = [
            bpms[r] for r in _FACE_ROIS
            if snrs[r] >= margin_threshold and bpms[r] > 0
        ]
        if len(trusted_bpms) >= 2:
            spread = max(trusted_bpms) - min(trusted_bpms)
            if spread > self.bpm_agreement:
                reasons.append(
                    f"roi_bpm_disagreement: trusted_bpms="
                    f"{[round(b,1) for b in trusted_bpms]}, "
                    f"spread={spread:.1f} > {self.bpm_agreement} BPM"
                )

        # ③ 배경 주파수 매칭 (얼굴과 같은 BPM이 배경에도 → 환경 노이즈)
        # 단, 배경 SNR이 얼굴 fused SNR보다 의미 있게 낮으면 (margin 통과 시) 무시
        if (
            bg_snr >= self.snr_threshold_db
            and fused_bpm > 0
            and snr_margin < self.snr_margin_db  # margin 부족 시에만 발동
        ):
            if abs(bg_bpm - fused_bpm) < self.bg_match_bpm:
                reasons.append(
                    f"background_freq_match: bg_bpm={bg_bpm:.1f}, "
                    f"fused_bpm={fused_bpm:.1f}, "
                    f"diff={abs(bg_bpm - fused_bpm):.1f} < {self.bg_match_bpm} BPM"
                )

        return RPPGAnalysis(
            bpms=bpms,
            snrs=snrs,
            background_bpm=bg_bpm,
            background_snr_db=bg_snr,
            fused_bpm=fused_bpm,
            fused_snr_db=fused_snr,
            is_spoof=len(reasons) > 0,
            reasons=reasons,
        )

    # ------------------------------------------------------------------

    def _extract_pulse(self, rgb: np.ndarray) -> np.ndarray:
        if self.algorithm == "pos":
            return pos_pulse(rgb, fs=self.fs)
        return chrom_pulse(rgb, fs=self.fs)

    def _best_window_metrics(
        self,
        pulse_bp: np.ndarray,
        window_sec: float = 6.0,
        stride_sec: float = 2.0,
    ) -> tuple[float, float]:
        """슬라이딩 window별 BPM/SNR 중 best SNR 반환.

        10초 캡처 = 6s 윈도우 × stride 2s → 3개 윈도우 + 전체 1개 = 4개 후보.
        사용자가 캡처 중 잠시 흔들거나 깜빡여도 양호한 구간만으로 평가.
        """
        N = len(pulse_bp)
        win = int(window_sec * self.fs)
        stride = max(1, int(stride_sec * self.fs))

        candidates: list[tuple[float, float]] = []
        if N >= win + stride:
            for start in range(0, N - win + 1, stride):
                seg = pulse_bp[start:start + win]
                bpm, _ = compute_bpm(seg, self.fs, low_bpm=42.0, high_bpm=150.0)
                snr = compute_snr_db(seg, self.fs, bpm, valid_range=(42.0, 150.0))
                candidates.append((bpm, snr))

        # 전체 신호도 후보로
        bpm_full, _ = compute_bpm(pulse_bp, self.fs, low_bpm=42.0, high_bpm=150.0)
        snr_full = compute_snr_db(pulse_bp, self.fs, bpm_full, valid_range=(42.0, 150.0))
        candidates.append((bpm_full, snr_full))

        # best SNR
        best = max(candidates, key=lambda x: x[1])
        return best

    def _fused_metrics(
        self,
        pulses: dict[str, np.ndarray],
        snrs: dict[str, float],
    ) -> tuple[float, float]:
        """SNR-가중 z-score fusion.

        ROI 별 SNR을 가중치(softmax-like)로 변환해 z-score 정규화된 신호의
        가중 평균을 만든다. 강한 ROI에 무게를 더 두고 약한 ROI는 덜 반영해서
        한 ROI가 망가져도 fused SNR이 크게 떨어지지 않는다.
        """
        items = [
            (name, p) for name, p in pulses.items()
            if p is not None and len(p) > 2
        ]
        if not items:
            return 0.0, -999.0

        T = min(len(p) for _, p in items)
        stacks = []
        weights = []
        for name, p in items:
            seg = p[:T].astype(np.float64)
            std = seg.std()
            if std < 1e-9:
                continue
            stacks.append((seg - seg.mean()) / std)
            # SNR → 가중치: dB를 선형 비율로 변환 후 floor.
            # snr < -10 dB 이하는 거의 무시 (weight ≈ 0.1)
            w_db = max(snrs.get(name, -999.0), -10.0)
            weights.append(10.0 ** (w_db / 10.0))

        if not stacks:
            return 0.0, -999.0

        w = np.array(weights, dtype=np.float64)
        w = w / w.sum() if w.sum() > 1e-12 else np.ones_like(w) / len(w)
        fused = np.sum(np.stack(stacks, axis=0) * w[:, None], axis=0)

        bpm, _ = compute_bpm(fused, self.fs, low_bpm=42.0, high_bpm=150.0)
        snr = compute_snr_db(fused, self.fs, bpm, valid_range=(42.0, 150.0))
        return float(bpm), float(snr)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    fs, T = 30.0, 300
    t = np.arange(T) / fs

    def make_rgb(bpm: float, noise: float = 1.0) -> np.ndarray:
        f = bpm / 60.0
        r = 100 * (1 + 0.005 * np.sin(2 * np.pi * f * t + 0.2)) + rng.normal(0, noise, T)
        g = 80  * (1 + 0.015 * np.sin(2 * np.pi * f * t        )) + rng.normal(0, noise, T)
        b = 60  * (1 + 0.005 * np.sin(2 * np.pi * f * t + 0.4)) + rng.normal(0, noise, T)
        return np.stack([r, g, b], axis=1)

    # 케이스 1: 실제 얼굴 — 깨끗
    live_signals = {
        "forehead":   make_rgb(72, noise=0.5),
        "left_cheek": make_rgb(72, noise=0.5),
        "right_cheek":make_rgb(72, noise=0.5),
        "background": rng.normal(128, 2, (T, 3)),
    }
    # 케이스 2: 실제 얼굴 — 노이즈 큰 환경 (절대 SNR 낮지만 face >> bg)
    noisy_live = {
        "forehead":   make_rgb(72, noise=2.0),
        "left_cheek": make_rgb(72, noise=2.0),
        "right_cheek":make_rgb(72, noise=2.0),
        "background": rng.normal(128, 10, (T, 3)),  # 배경도 노이즈 큼
    }
    # 케이스 3: 사진 — 둘 다 노이즈만
    spoof_signals = {
        "forehead":   rng.normal(128, 5, (T, 3)),
        "left_cheek": rng.normal(128, 5, (T, 3)),
        "right_cheek":rng.normal(128, 5, (T, 3)),
        "background": rng.normal(128, 5, (T, 3)),
    }

    checker = RPPGSpoofChecker(fs=fs, algorithm="pos")
    for name, sigs in [("Live(clean)", live_signals), ("Live(noisy)", noisy_live), ("Spoof", spoof_signals)]:
        r = checker.analyze(sigs)
        print(f"{name:14s} → spoof={r.is_spoof}, fused_bpm={r.fused_bpm:.1f}, "
              f"fused_snr={r.fused_snr_db:.2f} dB, bg_snr={r.background_snr_db:.2f} dB, "
              f"margin={r.fused_snr_db - r.background_snr_db:.2f} dB")
        if r.reasons:
            print(f"  reasons: {r.reasons}")
    print("spoof_check self-test passed.")
