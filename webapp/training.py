"""Moiré LBP+SVM 모델 학습 로직.

모듈 import 시점에는 무거운 의존성(mediapipe 등)을 불러오지 않는다.
모든 무거운 import 는 train_moire_model 함수 내부에서 지연 수행한다.
"""
from __future__ import annotations

import os
import glob


VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")


def _list_videos(folder: str) -> list[str]:
    files: list[str] = []
    for ext in VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def _sample_indices(n_total: int, n_want: int) -> list[int]:
    if n_total <= 0:
        return []
    if n_total <= n_want:
        return list(range(n_total))
    step = n_total / float(n_want)
    return [int(i * step) for i in range(n_want)]


def train_moire_model(
    data_root: str = "data",
    model_path: str = "models/moire_lbp_svm.pkl",
    frames_per_video: int = 8,
) -> dict:
    """data/live(=0), data/photo+data/screen(=1) 영상에서 얼굴 크롭을 모아
    MoireLBPSVMDetector 를 학습하고 저장한다.

    반환: {ok, n_live, n_spoof, train_accuracy, model_path, message}
    """
    # 지연 import (모듈 import 를 가볍게 유지)
    import numpy as np
    from liveness_pipeline import _read_video_frames
    from face_detector import FaceROIDetector, get_face_crop
    from moire.lbp_svm_detector import MoireLBPSVMDetector

    live_videos = _list_videos(os.path.join(data_root, "live"))
    spoof_videos = (
        _list_videos(os.path.join(data_root, "photo"))
        + _list_videos(os.path.join(data_root, "screen"))
    )

    if not live_videos or not spoof_videos:
        return {
            "ok": False,
            "n_live": 0,
            "n_spoof": 0,
            "train_accuracy": None,
            "model_path": model_path,
            "message": (
                "학습에 필요한 데이터가 부족합니다. "
                "실제 얼굴(live) 영상과 스푸핑(photo/screen) 영상이 "
                "각각 최소 1개씩 필요합니다."
            ),
        }

    detector = FaceROIDetector()
    crops: list[np.ndarray] = []
    labels: list[int] = []
    n_live = 0
    n_spoof = 0

    try:
        for path, label in (
            [(p, 0) for p in live_videos] + [(p, 1) for p in spoof_videos]
        ):
            try:
                frames = _read_video_frames(path)
            except Exception:
                continue
            if not frames:
                continue
            for idx in _sample_indices(len(frames), frames_per_video):
                frame = frames[idx]
                det = detector.detect(frame)
                if not det or "face_bbox" not in det:
                    continue
                crop = get_face_crop(frame, det["face_bbox"])
                if crop is None or crop.size == 0:
                    continue
                crops.append(crop)
                labels.append(label)
                if label == 0:
                    n_live += 1
                else:
                    n_spoof += 1
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            try:
                detector.close()
            except Exception:
                pass

    if n_live < 1 or n_spoof < 1:
        return {
            "ok": False,
            "n_live": n_live,
            "n_spoof": n_spoof,
            "train_accuracy": None,
            "model_path": model_path,
            "message": (
                "영상에서 얼굴이 검출된 샘플이 부족합니다 "
                f"(live={n_live}, spoof={n_spoof}). "
                "얼굴이 잘 보이는 영상으로 다시 수집해 주세요."
            ),
        }

    model = MoireLBPSVMDetector()
    model.fit(crops, labels)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)

    # 학습 정확도 (train accuracy — 동일 데이터 예측이므로 낙관적 수치)
    correct = 0
    total = 0
    for crop, label in zip(crops, labels):
        try:
            res = model.analyze_image(crop)
        except Exception:
            continue
        pred = int(bool(getattr(res, "is_spoof", False)))
        total += 1
        if pred == label:
            correct += 1
    train_accuracy = (correct / total) if total else None

    return {
        "ok": True,
        "n_live": n_live,
        "n_spoof": n_spoof,
        "train_accuracy": train_accuracy,
        "model_path": model_path,
        "message": "학습 완료. 이후 인증 시 학습 모델이 자동 적용됩니다.",
    }
