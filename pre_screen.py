"""pre_screen.py — rPPG 분석 전 환경 노이즈 사전 거부 모듈.

조명 급변 또는 큰 모션이 감지되면 reject해 rPPG의 false pulse를 방지한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreScreenResult:
    accepted: bool
    illumination_std: float   # 프레임 평균 밝기의 표준편차
    motion_magnitude: float   # 평균 optical flow magnitude
    reasons: list[str] = field(default_factory=list)
    # 한글 설명 사유 (UI 표시용). reasons와 1:1 대응.
    korean_reasons: list[str] = field(default_factory=list)


class PreScreener:
    """조명 급변·큰 모션 감지 시 프레임 시퀀스를 reject."""

    def __init__(
        self,
        illumination_std_threshold: float = 40.0,  # 밝기 std 허용 상한
        motion_threshold: float = 25.0,             # LK optical flow 평균 px/frame 허용 상한
        max_frames: int = 30,                       # stride 후 최대 분석 프레임 수
    ) -> None:
        self.illumination_std_threshold = illumination_std_threshold
        self.motion_threshold = motion_threshold
        self.max_frames = max_frames

        # LK optical flow 파라미터
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self._feature_params = dict(
            maxCorners=80,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )

    # ------------------------------------------------------------------
    def analyze(self, frames_bgr: list[np.ndarray]) -> PreScreenResult:
        """프레임 시퀀스 입력 → 거부 여부.

        1. 각 프레임 grayscale 평균 밝기 시계열 → std 계산
        2. 인접 프레임 간 Lucas-Kanade optical flow (sparse) 평균 magnitude
        3. 둘 중 하나라도 임계값 초과면 reject
        """
        if len(frames_bgr) < 2:
            return PreScreenResult(
                accepted=True,
                illumination_std=0.0,
                motion_magnitude=0.0,
                reasons=["too few frames to screen"],
            )

        # 프레임 수가 많으면 stride 샘플링으로 메모리 절약
        frames = self._sample_frames(frames_bgr)
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        illum_std = self._calc_illumination_std(grays)
        motion_mag = self._calc_motion_magnitude(grays)

        reasons: list[str] = []
        korean: list[str] = []
        if illum_std > self.illumination_std_threshold:
            reasons.append(
                f"illumination_std={illum_std:.1f} > {self.illumination_std_threshold}"
            )
            korean.append(
                f"조명이 너무 급격히 변합니다 — 안정된 조명에서 다시 시도하세요 "
                f"(수치: 밝기 변화 {illum_std:.0f} > 허용 {self.illumination_std_threshold:.0f})"
            )
        if motion_mag > self.motion_threshold:
            reasons.append(
                f"motion_magnitude={motion_mag:.2f} > {self.motion_threshold}"
            )
            korean.append(
                f"화면 움직임이 너무 큽니다 — 카메라를 정면으로 고정하세요 "
                f"(수치: 움직임 {motion_mag:.0f} > 허용 {self.motion_threshold:.0f}px/프레임)"
            )

        return PreScreenResult(
            accepted=len(reasons) == 0,
            illumination_std=float(illum_std),
            motion_magnitude=float(motion_mag),
            reasons=reasons,
            korean_reasons=korean,
        )

    # ------------------------------------------------------------------
    def _sample_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """프레임 수가 max_frames 초과 시 균등 stride 샘플링."""
        n = len(frames)
        if n <= self.max_frames:
            return frames
        step = n // self.max_frames
        return frames[::step][: self.max_frames]

    def _calc_illumination_std(self, grays: list[np.ndarray]) -> float:
        """각 프레임 평균 밝기 시계열의 표준편차."""
        means = np.array([g.mean() for g in grays], dtype=np.float32)
        return float(means.std())

    def _calc_motion_magnitude(self, grays: list[np.ndarray]) -> float:
        """인접 프레임 쌍에 걸쳐 LK optical flow 평균 magnitude를 구한다.

        특징점을 첫 프레임에서 한 번만 뽑아 재사용 — 소규모 PoC에 충분.
        """
        magnitudes: list[float] = []

        for prev_gray, next_gray in zip(grays[:-1], grays[1:]):
            pts = cv2.goodFeaturesToTrack(prev_gray, **self._feature_params)
            if pts is None or len(pts) == 0:
                continue

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, pts, None, **self._lk_params
            )
            if next_pts is None:
                continue

            good_prev = pts[status.ravel() == 1]
            good_next = next_pts[status.ravel() == 1]
            if len(good_prev) == 0:
                continue

            diff = good_next - good_prev          # shape (N, 1, 2)
            mag = np.linalg.norm(diff.reshape(-1, 2), axis=1)
            magnitudes.append(float(mag.mean()))

        return float(np.mean(magnitudes)) if magnitudes else 0.0


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 랜덤 프레임 30장으로 self-test (에러 없이 돌면 OK)
    rng = np.random.default_rng(0)
    dummy_frames = [
        rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(30)
    ]
    screener = PreScreener()
    result = screener.analyze(dummy_frames)
    print(f"[pre_screen self-test] accepted={result.accepted}, "
          f"illum_std={result.illumination_std:.2f}, "
          f"motion={result.motion_magnitude:.2f}")
    print("  reasons:", result.reasons or "none")
    print("PASS")
