from dataclasses import dataclass
from pathlib import Path
import numpy as np
import face_recognition

from .enroll import ENROLLED_DB_PATH, load_enrolled


@dataclass
class RecognitionResult:
    matched: bool
    name: str | None   # 매칭된 이름 또는 None
    distance: float    # 가장 가까운 encoding과의 distance
    threshold: float


class FaceRecognizer:
    def __init__(self, threshold: float = 0.6,
                 db_path: str | Path | None = None):
        """db_path None이면 enroll.ENROLLED_DB_PATH 사용. 로드는 lazy."""
        self.threshold = threshold
        self._db_path = Path(db_path) if db_path is not None else ENROLLED_DB_PATH
        self._db: dict[str, np.ndarray] | None = None

    def reload(self) -> None:
        """DB 다시 로드 (등록 후 호출)."""
        self._db = load_enrolled(self._db_path)

    def _get_db(self) -> dict[str, np.ndarray]:
        # lazy load: 첫 인식 시점에 DB를 읽음
        if self._db is None:
            self._db = load_enrolled(self._db_path)
        return self._db

    def recognize_image(self, image_rgb: np.ndarray) -> RecognitionResult:
        """RGB ndarray에서 첫 얼굴을 찾아 등록자와 매칭.

        등록자 0명이면 matched=False 반환 (에러 없음).
        """
        db = self._get_db()

        locations = face_recognition.face_locations(image_rgb)
        if not locations:
            return RecognitionResult(matched=False, name=None,
                                     distance=float("inf"), threshold=self.threshold)

        # 첫 번째 얼굴만 사용
        encodings = face_recognition.face_encodings(image_rgb, [locations[0]])
        if not encodings:
            return RecognitionResult(matched=False, name=None,
                                     distance=float("inf"), threshold=self.threshold)

        query_enc = encodings[0]

        if not db:
            # 등록자 없음 — 매칭 불가
            return RecognitionResult(matched=False, name=None,
                                     distance=float("inf"), threshold=self.threshold)

        names = list(db.keys())
        known_encs = np.array(list(db.values()))

        distances = face_recognition.face_distance(known_encs, query_enc)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        if best_dist < self.threshold:
            return RecognitionResult(matched=True, name=names[best_idx],
                                     distance=best_dist, threshold=self.threshold)
        return RecognitionResult(matched=False, name=None,
                                 distance=best_dist, threshold=self.threshold)

    def recognize_bgr(self, image_bgr: np.ndarray) -> RecognitionResult:
        """BGR ndarray 편의 함수 (cv2 프레임 직접 전달 시 사용)."""
        import cv2  # opencv는 BGR→RGB 변환에만 사용
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.recognize_image(image_rgb)


def recognize_face(image_rgb: np.ndarray,
                   threshold: float = 0.6,
                   db_path: str | Path | None = None) -> RecognitionResult:
    """One-shot 함수형 API (단발성 호출 시 사용)."""
    recognizer = FaceRecognizer(threshold=threshold, db_path=db_path)
    return recognizer.recognize_image(image_rgb)
