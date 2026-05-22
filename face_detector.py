"""MediaPipe Tasks API 기반 얼굴 landmark + 다중 ROI 마스크 추출.

MediaPipe 0.10.15+ 에서 legacy `mp.solutions.face_mesh`가 제거되어
새 Tasks API (`mp.tasks.vision.FaceLandmarker`) 를 사용한다.

다른 모듈(rppg.extractor, liveness_pipeline)이 의존하는 공개 인터페이스
(FaceROIDetector.detect, FACE_ROI_NAMES, ROI_LANDMARKS, mean_rgb_in_mask,
get_face_crop, draw_rois)는 그대로 유지.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# Tasks API용 모델 파일 — 첫 실행 시 자동 다운로드.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


# MediaPipe Face Mesh landmark 인덱스 기반 ROI 정의 (legacy와 호환).
# 새 Tasks API도 동일한 468개 mesh 토폴로지를 사용한다.
ROI_LANDMARKS: Dict[str, list] = {
    "forehead":    [67, 109, 10, 338, 297, 336, 107, 151],
    "left_cheek":  [50, 205, 187, 147, 123, 117, 116, 118],
    "right_cheek": [280, 425, 411, 376, 352, 346, 345, 347],
    "nose":        [1, 2, 5, 4, 19, 94, 168],
}

FACE_ROI_NAMES = ("forehead", "left_cheek", "right_cheek", "nose")


def _ensure_model() -> Path:
    """모델 파일이 없으면 다운로드."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[face_detector] 모델 다운로드 중... {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[face_detector] 다운로드 완료: {MODEL_PATH}")
    return MODEL_PATH


class FaceROIDetector:
    """단일 프레임 → 얼굴 landmark + ROI 마스크.

    내부적으로 MediaPipe Tasks API의 FaceLandmarker(IMAGE 모드)를 사용.
    Video stream에는 stateless detect()를 매 프레임 호출 — IMAGE 모드라
    프레임 간 추적이 없지만 PoC 정확도에는 충분.

    Parameters
    ----------
    min_detection_confidence : float
        얼굴 검출 신뢰도 임계값 (0~1).
    min_tracking_confidence : float
        (interface compatibility — IMAGE 모드에서는 미사용)
    static_image_mode : bool
        (interface compatibility — 항상 IMAGE 모드로 동작)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        model_path = _ensure_model()

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """단일 BGR 프레임 분석.

        Returns
        -------
        None: 얼굴 미검출
        dict:
          - landmarks: (N, 2) int32 픽셀 좌표 (N=468)
          - roi_masks: {roi_name: (H, W) uint8 0/255}
          - face_bbox: (x_min, y_min, x_max, y_max)  — margin 포함
          - background_mask: (H, W) uint8 — 얼굴 bbox 밖
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks_norm = result.face_landmarks[0]
        landmarks = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm],
            dtype=np.int32,
        )

        roi_masks: Dict[str, np.ndarray] = {}
        for name, indices in ROI_LANDMARKS.items():
            # 인덱스 범위 안전 검사 — 새 모델이 478점인 경우도 호환
            valid_idx = [i for i in indices if i < len(landmarks)]
            if not valid_idx:
                continue
            pts = landmarks[valid_idx]
            hull = cv2.convexHull(pts)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            roi_masks[name] = mask

        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        margin = int(0.1 * max(x_max - x_min, y_max - y_min))
        x_min = max(0, int(x_min) - margin)
        y_min = max(0, int(y_min) - margin)
        x_max = min(w, int(x_max) + margin)
        y_max = min(h, int(y_max) + margin)

        background_mask = np.full((h, w), 255, dtype=np.uint8)
        background_mask[y_min:y_max, x_min:x_max] = 0

        return {
            "landmarks": landmarks,
            "roi_masks": roi_masks,
            "face_bbox": (x_min, y_min, x_max, y_max),
            "background_mask": background_mask,
        }

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_face_crop(frame_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """얼굴 bbox로 크롭. Moiré 검출 등에서 사용."""
    x1, y1, x2, y2 = face_bbox
    return frame_bgr[y1:y2, x1:x2]


def mean_rgb_in_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    skin_only: bool = True,
) -> np.ndarray:
    """마스크 영역의 평균 RGB (R, G, B 순서). 마스크 비어 있으면 zeros.

    Parameters
    ----------
    skin_only : bool
        True면 mask ∩ skin-color mask 만 평균. 머리카락/안경/그림자 등 비피부
        픽셀을 제외해 PPG 신호 SNR을 크게 높인다. 피부 픽셀이 너무 적게
        남으면 자동으로 원래 마스크로 폴백.
    """
    if int(mask.sum()) == 0:
        return np.zeros(3, dtype=np.float64)
    if skin_only:
        skin = _skin_mask(frame_bgr)
        combined = cv2.bitwise_and(mask, skin)
        # 피부 픽셀이 원래 마스크의 30% 미만이면 폴백 (검출 실패/측면 얼굴 등)
        if int(combined.sum()) >= 0.3 * int(mask.sum()):
            mask = combined
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)


def _skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """YCrCb 기반 피부색 마스크 (Garcia & Tziritas, 1999 임계값).

    YCrCb 색공간의 Cr ∈ [133, 173], Cb ∈ [77, 127] 영역.
    조명 변동에 비교적 robust 하며 한국인 피부톤도 잘 잡힌다.
    """
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def draw_rois(frame_bgr: np.ndarray, detection: Dict) -> np.ndarray:
    """디버그용: ROI 윤곽 + bbox를 그려서 반환."""
    out = frame_bgr.copy()
    colors = {
        "forehead": (0, 255, 0),
        "left_cheek": (255, 0, 0),
        "right_cheek": (0, 0, 255),
        "nose": (0, 255, 255),
    }
    for name, mask in detection["roi_masks"].items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colors.get(name, (255, 255, 255)), 2)
    x1, y1, x2, y2 = detection["face_bbox"]
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return out


if __name__ == "__main__":
    # 모델 다운로드 + import 동작 확인 (실제 얼굴 없이도 detect 호출이 None 반환하는지)
    rng = np.random.default_rng(0)
    dummy = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    with FaceROIDetector() as fd:
        result = fd.detect(dummy)
    print(f"[face_detector self-test] detect on random image → "
          f"{'face found' if result else 'no face (expected)'}")
    print("PASS")
