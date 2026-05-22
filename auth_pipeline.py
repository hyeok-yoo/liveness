"""Liveness + Face Recognition 통합 인증 파이프라인.

흐름:
  1. LivenessPipeline.analyze_frames() — spoof 거부
  2. LIVE 통과 시 representative_frame으로 FaceRecognizer 매칭
  3. AuthResult 반환
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from liveness_pipeline import LivenessPipeline, LivenessResult
from face_auth import FaceRecognizer
from face_auth.recognize import RecognitionResult


@dataclass
class AuthResult:
    success: bool
    stage: str               # "spoof_blocked" | "no_match" | "match" | "no_face"
    name: Optional[str]      # 매칭된 사용자 이름 (success일 때만)
    liveness: LivenessResult
    recognition: Optional[RecognitionResult] = None
    message: str = ""


class AuthPipeline:
    """Liveness 통과 → face_recognition 매칭 → 최종 인증.

    Parameters
    ----------
    liveness : LivenessPipeline
        외부에서 구성한 liveness 파이프라인 인스턴스.
    recognizer : FaceRecognizer
        외부에서 구성한 얼굴 인식기. None이면 기본값으로 생성.
    """

    def __init__(
        self,
        liveness: Optional[LivenessPipeline] = None,
        recognizer: Optional[FaceRecognizer] = None,
    ) -> None:
        self.liveness = liveness if liveness is not None else LivenessPipeline()
        self.recognizer = recognizer if recognizer is not None else FaceRecognizer()

    def authenticate_frames(self, frames_bgr: list[np.ndarray]) -> AuthResult:
        """프레임 시퀀스로 인증.

        Returns
        -------
        AuthResult
            stage 값으로 GUI가 상태별 메시지를 결정.
        """
        liveness_res = self.liveness.analyze_frames(frames_bgr)

        if not liveness_res.is_live:
            return AuthResult(
                success=False,
                stage="spoof_blocked",
                name=None,
                liveness=liveness_res,
                message="스푸핑 감지: " + ", ".join(liveness_res.reasons[:3]),
            )

        if liveness_res.representative_frame is None:
            return AuthResult(
                success=False,
                stage="no_face",
                name=None,
                liveness=liveness_res,
                message="대표 얼굴 프레임 없음",
            )

        rec = self.recognizer.recognize_bgr(liveness_res.representative_frame)

        if rec.matched:
            return AuthResult(
                success=True,
                stage="match",
                name=rec.name,
                liveness=liveness_res,
                recognition=rec,
                message=f"환영합니다, {rec.name}",
            )

        return AuthResult(
            success=False,
            stage="no_match",
            name=None,
            liveness=liveness_res,
            recognition=rec,
            message=f"등록되지 않은 사용자 (distance={rec.distance:.3f})",
        )

    def authenticate_video_file(self, path: str) -> AuthResult:
        frames = self._read(path)
        return self.authenticate_frames(frames)

    @staticmethod
    def _read(path: str) -> list[np.ndarray]:
        from liveness_pipeline import _read_video_frames
        return _read_video_frames(path)

    def close(self) -> None:
        self.liveness.close()


if __name__ == "__main__":
    # face_recognition 라이브러리 미설치 환경에서도 import 자체는 통과해야 함
    print("auth_pipeline import OK")
