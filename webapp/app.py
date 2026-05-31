"""얼굴 라이브니스 검사 웹 제어판 (Flask 백엔드).

중요: 이 모듈은 mediapipe / face_recognition 없이도 `import webapp.app` 으로
깨끗하게 import 되어야 한다. 따라서 liveness_pipeline / auth_pipeline /
face_auth / face_detector 등 무거운 모듈은 **절대 모듈 최상단에서 import 하지 않고**
라우트 핸들러 또는 지연 로더(_Loader) 내부에서만 import 한다.
"""
from __future__ import annotations

import os
import time
import base64
import threading

import numpy as np
import cv2
from flask import Flask, jsonify, request, render_template


# 프로젝트 루트 (webapp/ 의 부모)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT_DIR, "data")
MODELS_ROOT = os.path.join(ROOT_DIR, "models")
MOIRE_MODEL_PATH = os.path.join(MODELS_ROOT, "moire_lbp_svm.pkl")

VALID_LABELS = ("live", "photo", "screen")

app = Flask(__name__)


# ---------------------------------------------------------------------------
# 지연 로더: 무거운 파이프라인/인식기 캐싱
# ---------------------------------------------------------------------------
class _Loader:
    """무거운 객체를 처음 요청 시점에 한 번만 생성하고 캐시한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recognizer = None  # FaceRecognizer 캐시

    # -- FaceRecognizer (등록자 임베딩 캐시) --
    def get_recognizer(self):
        with self._lock:
            if self._recognizer is None:
                from face_auth import FaceRecognizer
                self._recognizer = FaceRecognizer()
            return self._recognizer

    def reset_recognizer(self) -> None:
        """등록/삭제 후 임베딩 캐시를 무효화한다."""
        with self._lock:
            self._recognizer = None

    # -- 요청별 LivenessPipeline 구성 --
    def build_liveness(self, fs: float, rppg_decisive: bool):
        """주어진 fps/모드로 LivenessPipeline 을 새로 만든다 (PoC 허용).

        Moiré 모델이 학습돼 있으면 LBP 분기를 활성화한다.
        """
        from liveness_pipeline import LivenessPipeline

        kwargs = dict(
            fs=float(fs) if fs and fs > 0 else 15.0,
            rppg_algorithm="pos",
            min_face_frames=30,
            enable_pre_screen=True,
            enable_moire_fft=True,
            enable_moire_lbp=False,
            enable_rppg=True,
            rppg_decisive=rppg_decisive,
            lbp_model_path=None,
        )
        if os.path.exists(MOIRE_MODEL_PATH):
            kwargs["enable_moire_lbp"] = True
            kwargs["lbp_model_path"] = MOIRE_MODEL_PATH
        return LivenessPipeline(**kwargs)

    def build_auth(self, liveness):
        from auth_pipeline import AuthPipeline
        return AuthPipeline(liveness=liveness, recognizer=self.get_recognizer())

    def pipeline_loaded(self) -> bool:
        return self._recognizer is not None


_loader = _Loader()


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def decode_frames(data_urls) -> list:
    """data:image/jpeg;base64,... 형태의 dataURL 리스트를 BGR ndarray 리스트로 디코드.

    cv2.imdecode 는 BGR 을 반환하므로 파이프라인에 바로 사용 가능. None 은 건너뜀.
    """
    frames = []
    if not data_urls:
        return frames
    for url in data_urls:
        if not url or not isinstance(url, str):
            continue
        b64 = url.split(",", 1)[1] if "," in url else url
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        frames.append(img)
    return frames


def _count_videos(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    return sum(
        1
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )


def _liveness_details(res) -> dict:
    """LivenessResult 에서 상세 수치를 안전하게 추출."""
    moire = getattr(res, "moire_fft", None)
    rppg = getattr(res, "rppg", None)
    return {
        "frames_total": getattr(res, "frames_total", 0),
        "frames_with_face": getattr(res, "frames_with_face", 0),
        "is_live": bool(getattr(res, "is_live", False)),
        "moire_score": getattr(moire, "score", None) if moire else None,
        "moire_threshold": getattr(moire, "threshold", None) if moire else None,
        "fused_bpm": getattr(rppg, "fused_bpm", None) if rppg else None,
        "fused_snr_db": getattr(rppg, "fused_snr_db", None) if rppg else None,
        "background_snr_db": getattr(rppg, "background_snr_db", None) if rppg else None,
    }


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """상태 조회. 무거운 파이프라인을 절대 생성하지 않는다 (500 방지)."""
    enrolled = []
    try:
        from face_auth import list_enrolled
        enrolled = list_enrolled()
    except Exception:
        enrolled = []
    return jsonify(
        {
            "pipeline_loaded": _loader.pipeline_loaded(),
            "moire_model_trained": os.path.exists(MOIRE_MODEL_PATH),
            "enrolled": enrolled,
            "sample_counts": {
                "live": _count_videos(os.path.join(DATA_ROOT, "live")),
                "photo": _count_videos(os.path.join(DATA_ROOT, "photo")),
                "screen": _count_videos(os.path.join(DATA_ROOT, "screen")),
            },
        }
    )


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    image = data.get("image")
    if not name:
        return jsonify({"ok": False, "message": "이름을 입력해 주세요."})
    frames = decode_frames([image] if image else [])
    if not frames:
        return jsonify({"ok": False, "message": "이미지를 읽을 수 없습니다."})
    # BGR → RGB (enroll_face_from_array 는 RGB 입력)
    rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    try:
        from face_auth import enroll_face_from_array
        enroll_face_from_array(name, rgb)
    except ValueError:
        return jsonify(
            {"ok": False, "message": "얼굴을 찾을 수 없습니다. 정면을 향해 다시 시도해 주세요."}
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": f"등록 실패: {exc}"})
    _loader.reset_recognizer()
    return jsonify({"ok": True, "message": f"'{name}' 등록 완료."})


@app.route("/api/enrolled")
def api_enrolled():
    try:
        from face_auth import list_enrolled
        return jsonify({"enrolled": list_enrolled()})
    except Exception:
        return jsonify({"enrolled": []})


@app.route("/api/remove_enrolled", methods=["POST"])
def api_remove_enrolled():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False})
    try:
        from face_auth import remove_enrolled
        ok = bool(remove_enrolled(name))
    except Exception:
        ok = False
    if ok:
        _loader.reset_recognizer()
    return jsonify({"ok": ok})


@app.route("/api/authenticate", methods=["POST"])
def api_authenticate():
    data = request.get_json(silent=True) or {}
    frames = decode_frames(data.get("frames"))
    fps = data.get("fps") or 15.0
    demo_mode = bool(data.get("demo_mode"))
    if not frames:
        return jsonify(
            {
                "success": False,
                "stage": "no_face",
                "name": None,
                "message": "프레임을 받지 못했습니다.",
                "korean_reasons": ["수신된 영상 프레임이 없습니다."],
                "details": {},
            }
        )

    liveness = _loader.build_liveness(fs=fps, rppg_decisive=(not demo_mode))
    auth = _loader.build_auth(liveness)
    try:
        result = auth.authenticate_frames(frames)
    finally:
        try:
            liveness.close()
        except Exception:
            pass

    lres = getattr(result, "liveness", None)
    korean_reasons = list(getattr(lres, "korean_reasons", [])) if lres else []
    details = _liveness_details(lres) if lres else {}
    return jsonify(
        {
            "success": bool(getattr(result, "success", False)),
            "stage": getattr(result, "stage", "no_face"),
            "name": getattr(result, "name", None),
            "message": getattr(result, "message", ""),
            "korean_reasons": korean_reasons,
            "details": details,
        }
    )


@app.route("/api/liveness", methods=["POST"])
def api_liveness():
    data = request.get_json(silent=True) or {}
    frames = decode_frames(data.get("frames"))
    fps = data.get("fps") or 15.0
    demo_mode = bool(data.get("demo_mode"))
    if not frames:
        return jsonify(
            {
                "is_live": False,
                "korean_reasons": ["수신된 영상 프레임이 없습니다."],
                "details": {},
            }
        )

    liveness = _loader.build_liveness(fs=fps, rppg_decisive=(not demo_mode))
    try:
        res = liveness.analyze_frames(frames)
    finally:
        try:
            liveness.close()
        except Exception:
            pass

    return jsonify(
        {
            "is_live": bool(getattr(res, "is_live", False)),
            "korean_reasons": list(getattr(res, "korean_reasons", [])),
            "details": _liveness_details(res),
        }
    )


@app.route("/api/collect_sample", methods=["POST"])
def api_collect_sample():
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()
    if label not in VALID_LABELS:
        return jsonify(
            {"ok": False, "message": f"라벨은 {VALID_LABELS} 중 하나여야 합니다."}
        )
    frames = decode_frames(data.get("frames"))
    if not frames:
        return jsonify({"ok": False, "message": "녹화된 프레임이 없습니다."})
    fps = float(data.get("fps") or 15.0)
    if fps <= 0:
        fps = 15.0

    folder = os.path.join(DATA_ROOT, label)
    os.makedirs(folder, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(folder, f"sample_{ts}.mp4")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        return jsonify({"ok": False, "message": "동영상 저장기를 열 수 없습니다."})
    n = 0
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)
        n += 1
    writer.release()

    return jsonify(
        {
            "ok": True,
            "path": out_path,
            "n_frames": n,
            "sample_counts": {
                "live": _count_videos(os.path.join(DATA_ROOT, "live")),
                "photo": _count_videos(os.path.join(DATA_ROOT, "photo")),
                "screen": _count_videos(os.path.join(DATA_ROOT, "screen")),
            },
        }
    )


@app.route("/api/train_moire", methods=["POST"])
def api_train_moire():
    from webapp.training import train_moire_model

    n_live = _count_videos(os.path.join(DATA_ROOT, "live"))
    n_spoof = _count_videos(os.path.join(DATA_ROOT, "photo")) + _count_videos(
        os.path.join(DATA_ROOT, "screen")
    )
    if n_live < 1 or n_spoof < 1:
        return jsonify(
            {
                "ok": False,
                "message": (
                    "데이터가 부족합니다. 실제 얼굴(live) 영상과 "
                    "스푸핑(photo/screen) 영상이 각각 최소 1개씩 필요합니다."
                ),
                "n_live": n_live,
                "n_spoof": n_spoof,
                "train_accuracy": None,
            }
        )

    try:
        res = train_moire_model(
            data_root=DATA_ROOT, model_path=MOIRE_MODEL_PATH
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": f"학습 실패: {exc}"})

    warn = ""
    if res.get("ok") and (res.get("n_live", 0) < 5 or res.get("n_spoof", 0) < 5):
        warn = " (권장: 각 클래스 5개 이상 — 더 많은 데이터로 정확도가 올라갑니다.)"

    return jsonify(
        {
            "ok": res.get("ok", False),
            "message": (res.get("message") or "") + warn,
            "n_live": res.get("n_live", 0),
            "n_spoof": res.get("n_spoof", 0),
            "train_accuracy": res.get("train_accuracy"),
        }
    )


def main() -> None:
    url = "http://127.0.0.1:5000"
    print("=" * 60)
    print("얼굴 라이브니스 검사 제어판이 실행 중입니다.")
    print(f"브라우저에서 다음 주소를 열어 주세요: {url}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
