"""moire/fft_detector.py — 2D FFT 기반 Moiré 패턴 검출.

화면 재촬영(spoof) 시 픽셀 그리드 간섭으로 mid-high 주파수 대역 에너지가 증가하는
원리를 이용. 학습 데이터 없이 임계값만으로 동작한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MoireFFTResult:
    is_spoof: bool
    score: float      # mid-high band 에너지 비율 (0 ~ 1)
    threshold: float


class MoireFFTDetector:
    """얼굴 크롭 이미지의 주파수 스펙트럼으로 Moiré 판정.

    학습 불필요 — threshold 하나만 튜닝하면 된다.
    """

    def __init__(
        self,
        patch_size: int = 256,    # 얼굴 영역 리사이즈 크기 (2의 거듭제곱 권장)
        low_band: float = 0.25,   # 정규화 주파수 (0~0.5), 링 내경
        high_band: float = 0.45,  # 링 외경
        threshold: float = 0.18,  # 이 값 이상이면 spoof; 데이터로 튜닝 필요
    ) -> None:
        self.patch_size = patch_size
        self.low_band = low_band
        self.high_band = high_band
        self.threshold = threshold

        # 링 마스크를 한 번만 생성해 재사용 (patch_size 고정 가정)
        self._ring_mask = self._build_ring_mask(patch_size, low_band, high_band)
        # Hann 윈도우: border artifact 억제 (경계 불연속 → spurious 고주파 방지)
        self._window = np.outer(np.hanning(patch_size), np.hanning(patch_size))

    # ------------------------------------------------------------------
    def analyze_image(self, face_crop_bgr: np.ndarray) -> MoireFFTResult:
        """단일 얼굴 이미지 분석."""
        score = self._compute_score(face_crop_bgr)
        return MoireFFTResult(
            is_spoof=score > self.threshold,
            score=score,
            threshold=self.threshold,
        )

    def analyze_video(self, face_crops: list[np.ndarray]) -> MoireFFTResult:
        """여러 프레임의 score를 평균 — 단일 프레임보다 안정적."""
        if not face_crops:
            return MoireFFTResult(is_spoof=False, score=0.0, threshold=self.threshold)
        scores = [self._compute_score(crop) for crop in face_crops]
        mean_score = float(np.mean(scores))
        return MoireFFTResult(
            is_spoof=mean_score > self.threshold,
            score=mean_score,
            threshold=self.threshold,
        )

    # ------------------------------------------------------------------
    def _compute_score(self, face_crop_bgr: np.ndarray) -> float:
        """Mid-high 대역 에너지 비율 계산.

        1. grayscale + resize → (N, N)
        2. 평균 제거 + Hann 윈도우 적용
        3. 2D FFT → 진폭 스펙트럼 → log(1 + |F|)
        4. 중심 기준 링 영역 에너지 / 전체 에너지
        """
        N = self.patch_size
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (N, N))

        # 평균 제거 후 윈도우 적용 (DC 성분이 스펙트럼 전체를 지배하는 현상 방지)
        gray -= gray.mean()
        windowed = gray * self._window

        spectrum = np.fft.fftshift(np.abs(np.fft.fft2(windowed)))
        log_spectrum = np.log1p(spectrum)  # log scale로 동적 범위 압축

        total_energy = log_spectrum.sum()
        if total_energy == 0.0:
            return 0.0

        mid_high_energy = (log_spectrum * self._ring_mask).sum()
        return float(mid_high_energy / total_energy)

    @staticmethod
    def _build_ring_mask(N: int, low_band: float, high_band: float) -> np.ndarray:
        """중심 기준 annular(링) 마스크 생성. 반경 단위는 정규화 주파수."""
        cy, cx = N // 2, N // 2
        # 좌표 그리드: 중심이 0
        v, u = np.mgrid[-cy : N - cy, -cx : N - cx]
        # 반경을 정규화 주파수로 환산 (최대 반경 = N/2)
        r = np.sqrt(u**2 + v**2) / (N / 2)
        mask = ((r >= low_band) & (r <= high_band)).astype(np.float32)
        return mask


# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dummy = rng.integers(0, 255, (200, 180, 3), dtype=np.uint8)

    detector = MoireFFTDetector()
    result = detector.analyze_image(dummy)
    print(f"[fft_detector self-test] is_spoof={result.is_spoof}, "
          f"score={result.score:.4f}, threshold={result.threshold}")

    frames = [rng.integers(0, 255, (200, 180, 3), dtype=np.uint8) for _ in range(5)]
    result_v = detector.analyze_video(frames)
    print(f"  video result: is_spoof={result_v.is_spoof}, score={result_v.score:.4f}")
    print("PASS")
