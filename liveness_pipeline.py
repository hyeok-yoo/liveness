"""Liveness 통합 파이프라인.

Pre-screen → Moiré (FFT, 선택적 LBP+SVM) → rPPG (POS or CHROM) 순서로 검사.
하나라도 spoof 신호를 내면 LIVE=False (OR 로직 — 보안 측면 보수적).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from face_detector import FaceROIDetector, get_face_crop
from pre_screen import PreScreener, PreScreenResult
from moire.fft_detector import MoireFFTDetector, MoireFFTResult
from moire.lbp_svm_detector import MoireLBPSVMDetector, MoireLBPResult
from rppg.extractor import RGBTimeSeriesExtractor
from rppg.spoof_check import RPPGSpoofChecker, RPPGAnalysis


@dataclass
class LivenessResult:
    is_live: bool
    reasons: list[str] = field(default_factory=list)
    pre_screen: Optional[PreScreenResult] = None
    moire_fft: Optional[MoireFFTResult] = None
    moire_lbp: Optional[MoireLBPResult] = None
    rppg: Optional[RPPGAnalysis] = None
    frames_total: int = 0
    frames_with_face: int = 0
    # 얼굴이 확실히 잡힌 대표 프레임 (face_recognition에서 사용). BGR.
    representative_frame: Optional[np.ndarray] = None


class LivenessPipeline:
    """Liveness 검사 오케스트레이션.

    각 검사기는 생성자에서 enable_* 플래그로 켜고 끌 수 있다 (ablation 평가용).

    Parameters
    ----------
    fs : float
        샘플링 주파수 (fps).
    rppg_algorithm : str
        "pos" 또는 "chrom".
    min_face_frames : int
        최소 얼굴 검출 프레임 수. 미달 시 자동 spoof 처리.
    """

    def __init__(
        self,
        fs: float = 30.0,
        rppg_algorithm: str = "pos",
        min_face_frames: int = 30,
        enable_pre_screen: bool = True,
        enable_moire_fft: bool = True,
        enable_moire_lbp: bool = False,
        enable_rppg: bool = True,
        rppg_decisive: bool = True,
        lbp_model_path: Optional[str] = None,
        # 검사기 임계값 — 데이터 튜닝 시 노출
        moire_fft_threshold: float = 0.18,
        rppg_snr_threshold_db: float = 3.0,
        rppg_snr_margin_db: float = 1.0,
        rppg_bpm_agreement: float = 20.0,
        rppg_bg_match_bpm: float = 5.0,
        prescreen_illumination_std: float = 40.0,
        prescreen_motion: float = 25.0,
    ) -> None:
        self.fs = fs
        self.min_face_frames = min_face_frames
        self.rppg_decisive = rppg_decisive

        self.detector = FaceROIDetector()

        self.pre_screener: Optional[PreScreener] = (
            PreScreener(
                illumination_std_threshold=prescreen_illumination_std,
                motion_threshold=prescreen_motion,
            )
            if enable_pre_screen
            else None
        )
        self.moire_fft: Optional[MoireFFTDetector] = (
            MoireFFTDetector(threshold=moire_fft_threshold) if enable_moire_fft else None
        )
        self.moire_lbp: Optional[MoireLBPSVMDetector] = None
        if enable_moire_lbp and lbp_model_path:
            self.moire_lbp = MoireLBPSVMDetector.load(lbp_model_path)

        self.rppg_checker: Optional[RPPGSpoofChecker] = None
        if enable_rppg:
            self.rppg_checker = RPPGSpoofChecker(
                fs=fs,
                snr_threshold_db=rppg_snr_threshold_db,
                snr_margin_db=rppg_snr_margin_db,
                bpm_agreement=rppg_bpm_agreement,
                bg_match_bpm=rppg_bg_match_bpm,
                algorithm=rppg_algorithm,
            )

    # ------------------------------------------------------------------

    def analyze_frames(self, frames_bgr: list[np.ndarray]) -> LivenessResult:
        """프레임 시퀀스(BGR ndarray list) → LivenessResult.

        호출자가 카메라/영상 디코딩을 책임지고, 여기서는 검사만 수행.
        """
        result = LivenessResult(is_live=False, frames_total=len(frames_bgr))

        if len(frames_bgr) < 2:
            result.reasons.append("not_enough_frames")
            return result

        # 1) Pre-screen — 조명 급변/큰 모션
        if self.pre_screener is not None:
            result.pre_screen = self.pre_screener.analyze(frames_bgr)
            if not result.pre_screen.accepted:
                result.reasons.extend(f"prescreen:{r}" for r in result.pre_screen.reasons)
                return result

        # 2) 통합 추출: 얼굴 검출 + RGB 시계열 + face crop을 단일 패스로
        extractor = RGBTimeSeriesExtractor(detector=self.detector)
        face_crops: list[np.ndarray] = []
        for frame in frames_bgr:
            ok, detection = extractor.process_frame_with_detection(frame)
            if not ok or detection is None:
                continue
            crop = get_face_crop(frame, detection["face_bbox"])
            if crop.size == 0:
                continue
            face_crops.append(crop)
            if result.representative_frame is None:
                # 첫 얼굴 검출 프레임을 face_recognition용 대표로 보관
                result.representative_frame = frame.copy()

        result.frames_with_face = extractor.frame_count_with_face

        if result.frames_with_face < self.min_face_frames:
            result.reasons.append(
                f"too_few_face_frames: {result.frames_with_face} < {self.min_face_frames}"
            )
            return result

        # 3) Moiré FFT — 얼굴 crops에서 mid-high 주파수 에너지
        if self.moire_fft is not None and face_crops:
            result.moire_fft = self.moire_fft.analyze_video(face_crops)
            if result.moire_fft.is_spoof:
                result.reasons.append(
                    f"moire_fft: score={result.moire_fft.score:.3f}"
                    f" > {result.moire_fft.threshold}"
                )
                # 보안 보수적: 일찍 거부
                return result

        # 4) Moiré LBP+SVM (선택)
        if self.moire_lbp is not None and face_crops:
            # 대표 프레임 하나만 사용 (학습 모델 추론 비용 줄이기)
            lbp_res = self.moire_lbp.analyze_image(face_crops[len(face_crops) // 2])
            result.moire_lbp = lbp_res
            if lbp_res.is_spoof:
                result.reasons.append(f"moire_lbp: score={lbp_res.score:.3f}")
                return result

        # 5) rPPG spoof check — 핵심 false positive 완화 레이어
        if self.rppg_checker is not None:
            signals = extractor.get_signals(interpolate=True)
            result.rppg = self.rppg_checker.analyze(signals)
            if result.rppg.is_spoof:
                # decisive 모드: rPPG가 spoof면 즉시 거부
                # non-decisive 모드(발표용): pre-screen + moire가 이미 통과했으므로
                #   사진/영상 위협은 다른 신호로 막힌 상태 → rPPG 약함은 경고만 표시하고 통과.
                if self.rppg_decisive:
                    result.reasons.extend(f"rppg:{r}" for r in result.rppg.reasons)
                    return result
                else:
                    result.reasons.extend(f"rppg_warn:{r}" for r in result.rppg.reasons)

        # 모든 검사 통과
        result.is_live = True
        return result

    # ------------------------------------------------------------------

    def analyze_video_file(self, path: str) -> LivenessResult:
        """동영상 파일 경로로부터 분석 (편의 함수)."""
        frames = _read_video_frames(path)
        return self.analyze_frames(frames)

    def close(self) -> None:
        self.detector.close()


# ----------------------------------------------------------------------


def _read_video_frames(path: str) -> list[np.ndarray]:
    """비디오 파일을 BGR 프레임 리스트로 디코딩."""
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    if not cap.isOpened():
        return frames
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 더미 프레임으로 self-test — face_detector가 얼굴을 못 잡아도 에러 없이 도는지만 확인
    rng = np.random.default_rng(0)
    dummy_frames = [
        rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(60)
    ]
    pipeline = LivenessPipeline(fs=30.0, enable_moire_lbp=False)
    res = pipeline.analyze_frames(dummy_frames)
    print(f"[liveness self-test] is_live={res.is_live}")
    print(f"  reasons: {res.reasons}")
    print(f"  frames: total={res.frames_total} with_face={res.frames_with_face}")
    pipeline.close()
    print("PASS")
