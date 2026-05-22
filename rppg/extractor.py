"""RGB 시계열 추출기. 각 프레임에서 얼굴 ROI 별 평균 RGB를 누적한다."""
from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np

from face_detector import FaceROIDetector, mean_rgb_in_mask, FACE_ROI_NAMES

_ALL_ROI_NAMES = list(FACE_ROI_NAMES) + ["background"]


class RGBTimeSeriesExtractor:
    """프레임 스트림 → ROI 별 RGB 시계열.

    Parameters
    ----------
    detector : FaceROIDetector, optional
        외부에서 생성한 검출기. None이면 내부 생성.
    """

    def __init__(self, detector: Optional[FaceROIDetector] = None) -> None:
        self._detector = detector if detector is not None else FaceROIDetector()
        self._buffers: dict[str, list[np.ndarray]] = {name: [] for name in _ALL_ROI_NAMES}
        self._frame_count = 0
        self._face_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> bool:
        """한 프레임을 처리해 내부 버퍼에 누적한다.

        Returns
        -------
        bool
            얼굴이 검출되면 True, 아니면 False (NaN 행 추가).
        """
        ok, _ = self.process_frame_with_detection(frame_bgr)
        return ok

    def process_frame_with_detection(
        self, frame_bgr: np.ndarray
    ) -> Tuple[bool, Optional[Dict]]:
        """process_frame + detection dict 반환.

        파이프라인 통합 시 Moiré용 face crop을 위해 detection이 필요하지만,
        중복 detect 호출은 피하고자 한 번에 반환한다.

        Returns
        -------
        tuple
            (face_detected, detection_dict_or_None)
        """
        self._frame_count += 1
        detection = self._detector.detect(frame_bgr)

        if detection is None:
            nan_row = np.full(3, np.nan)
            for name in _ALL_ROI_NAMES:
                self._buffers[name].append(nan_row.copy())
            return False, None

        self._face_count += 1
        for name in FACE_ROI_NAMES:
            mask = detection["roi_masks"][name]
            # 얼굴 ROI는 skin-pixel masking으로 머리카락/안경/그림자 제외
            rgb = mean_rgb_in_mask(frame_bgr, mask, skin_only=True)
            self._buffers[name].append(rgb)

        # 배경은 피부 마스킹 끔 (배경에 피부 없음 → 평균이 0이 됨)
        bg_rgb = mean_rgb_in_mask(frame_bgr, detection["background_mask"], skin_only=False)
        self._buffers["background"].append(bg_rgb)
        return True, detection

    def get_signals(self, interpolate: bool = True) -> dict[str, np.ndarray]:
        """ROI 별 RGB 시계열 반환.

        Parameters
        ----------
        interpolate : bool
            True일 때 짧은 NaN 구간(연속 NaN)을 선형 보간으로 채운다.

        Returns
        -------
        dict[str, np.ndarray]
            {roi_name: (T, 3)} float64 배열.
        """
        result: dict[str, np.ndarray] = {}
        for name, buf in self._buffers.items():
            if not buf:
                result[name] = np.empty((0, 3), dtype=np.float64)
                continue
            arr = np.array(buf, dtype=np.float64)  # (T, 3)
            if interpolate:
                arr = _interpolate_nan_rows(arr)
            result[name] = arr
        return result

    def reset(self) -> None:
        """버퍼를 비운다."""
        self._buffers = {name: [] for name in _ALL_ROI_NAMES}
        self._frame_count = 0
        self._face_count = 0

    def __len__(self) -> int:
        return self._frame_count

    @property
    def frame_count_with_face(self) -> int:
        """얼굴이 검출된 프레임 수."""
        return self._face_count


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _interpolate_nan_rows(arr: np.ndarray) -> np.ndarray:
    """(T, 3) 배열에서 NaN 행을 채널 별 선형 보간으로 채운다."""
    T = arr.shape[0]
    t = np.arange(T, dtype=np.float64)
    out = arr.copy()
    for c in range(3):
        col = out[:, c]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        valid = ~nan_mask
        if valid.sum() < 2:
            # 보간 불가 → 0으로 채움
            col[nan_mask] = 0.0
        else:
            col[nan_mask] = np.interp(t[nan_mask], t[valid], col[valid])
        out[:, c] = col
    return out


if __name__ == "__main__":
    import numpy as np

    # 랜덤 신호로 self-test (face_detector 없이 직접 버퍼에 삽입)
    ext = RGBTimeSeriesExtractor.__new__(RGBTimeSeriesExtractor)
    ext._buffers = {name: [] for name in _ALL_ROI_NAMES}
    ext._frame_count = 0
    ext._face_count = 0

    rng = np.random.default_rng(0)
    for i in range(100):
        row = rng.uniform(80, 200, size=3) if i % 10 != 5 else np.full(3, np.nan)
        for name in _ALL_ROI_NAMES:
            ext._buffers[name].append(row.copy())
        ext._frame_count += 1
        if not np.isnan(row[0]):
            ext._face_count += 1

    sigs = ext.get_signals(interpolate=True)
    for k, v in sigs.items():
        assert not np.isnan(v).any(), f"NaN found in {k}"
        assert v.shape == (100, 3)
    print("extractor self-test passed:", {k: v.shape for k, v in sigs.items()})
