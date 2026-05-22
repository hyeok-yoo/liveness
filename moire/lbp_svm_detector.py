"""moire/lbp_svm_detector.py — LBP 히스토그램 + SVM Moiré 검출.

소규모 데이터(클래스당 10장)에서도 잘 동작하는 텍스처 기반 분류기.
fit() 호출 전에는 analyze_image()가 NotFittedError를 발생시킨다.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class MoireLBPResult:
    is_spoof: bool
    score: float   # spoof 확률 (0 ~ 1)


class MoireLBPSVMDetector:
    """LBP 텍스처 히스토그램 + RBF SVM 분류기.

    Moiré 패턴은 규칙적인 고주파 텍스처를 형성하므로 LBP로 포착 가능.
    데이터 수집 후 fit()을 호출해 학습한다.
    """

    def __init__(
        self,
        patch_size: int = 128,
        lbp_P: int = 8,
        lbp_R: int = 1,
        lbp_method: str = "uniform",
    ) -> None:
        self.patch_size = patch_size
        self.lbp_P = lbp_P
        self.lbp_R = lbp_R
        self.lbp_method = lbp_method

        # uniform LBP의 bin 수: P + 2 (P개 uniform 패턴 + non-uniform 1 + 패딩)
        self._n_bins: int = lbp_P + 2

        # 학습 전에는 None; analyze_image 호출 시 NotFittedError 발생
        self.classifier: Pipeline | None = None

    # ------------------------------------------------------------------
    def extract_features(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """얼굴 크롭 → 정규화된 LBP 히스토그램 (shape: (n_bins,))."""
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.patch_size, self.patch_size))

        lbp = local_binary_pattern(
            gray, self.lbp_P, self.lbp_R, method=self.lbp_method
        )
        # density=True로 정규화 → 이미지 크기 불변
        hist, _ = np.histogram(
            lbp.ravel(), bins=self._n_bins, range=(0, self._n_bins), density=True
        )
        return hist.astype(np.float32)

    def fit(self, face_crops: list[np.ndarray], labels: list[int]) -> None:
        """labels: 0=live, 1=spoof. SVM(RBF 커널) 학습.

        Pipeline에 StandardScaler를 포함시켜 특징 스케일 의존성 제거.
        """
        if len(face_crops) == 0:
            raise ValueError("학습 데이터가 없습니다.")

        X = np.array([self.extract_features(crop) for crop in face_crops])
        y = np.array(labels, dtype=np.int32)

        # RBF SVM + StandardScaler Pipeline — 소규모 데이터에 과적합 억제
        self.classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ])
        self.classifier.fit(X, y)

    def analyze_image(self, face_crop_bgr: np.ndarray) -> MoireLBPResult:
        """학습된 모델로 단일 이미지 예측.

        fit()이 호출되지 않았으면 NotFittedError를 발생시킨다.
        """
        if self.classifier is None:
            raise NotFittedError(
                "MoireLBPSVMDetector가 아직 학습되지 않았습니다. "
                "fit(face_crops, labels)를 먼저 호출하세요."
            )
        feat = self.extract_features(face_crop_bgr).reshape(1, -1)
        proba = self.classifier.predict_proba(feat)[0]
        # classes_ 순서가 [0, 1] 보장 안 될 수 있으므로 인덱스 명시
        spoof_idx = list(self.classifier.classes_).index(1)
        spoof_prob = float(proba[spoof_idx])
        return MoireLBPResult(is_spoof=spoof_prob >= 0.5, score=spoof_prob)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """모델(Pipeline) + 메타정보를 pickle로 저장."""
        payload = {
            "classifier": self.classifier,
            "patch_size": self.patch_size,
            "lbp_P": self.lbp_P,
            "lbp_R": self.lbp_R,
            "lbp_method": self.lbp_method,
            "n_bins": self._n_bins,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "MoireLBPSVMDetector":
        """저장된 pickle에서 모델 복원."""
        if not Path(path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        detector = cls(
            patch_size=payload["patch_size"],
            lbp_P=payload["lbp_P"],
            lbp_R=payload["lbp_R"],
            lbp_method=payload["lbp_method"],
        )
        detector.classifier = payload["classifier"]
        detector._n_bins = payload["n_bins"]
        return detector


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import os

    rng = np.random.default_rng(7)
    detector = MoireLBPSVMDetector()

    # 1) fit 전 NotFittedError 확인
    dummy = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    try:
        detector.analyze_image(dummy)
        print("FAIL — NotFittedError가 발생해야 합니다")
    except NotFittedError as e:
        print(f"[lbp_svm self-test] NotFittedError 정상: {e}")

    # 2) 더미 데이터로 학습 후 예측
    crops = [rng.integers(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    detector.fit(crops, labels)

    result = detector.analyze_image(dummy)
    print(f"  analyze_image: is_spoof={result.is_spoof}, score={result.score:.4f}")

    # 3) save/load 왕복 테스트
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        detector.save(tmp_path)
        loaded = MoireLBPSVMDetector.load(tmp_path)
        result2 = loaded.analyze_image(dummy)
        print(f"  loaded model: is_spoof={result2.is_spoof}, score={result2.score:.4f}")
    finally:
        os.unlink(tmp_path)

    print("PASS")
