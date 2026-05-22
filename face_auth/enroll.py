from pathlib import Path
import pickle
import numpy as np
import face_recognition

ENROLLED_DB_PATH = Path("data/enrolled/db.pkl")


def _load_db(db_path: Path) -> dict[str, np.ndarray]:
    if db_path.exists():
        with open(db_path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_db(db: dict[str, np.ndarray], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "wb") as f:
        pickle.dump(db, f)


def enroll_face(name: str, image_path: str | Path,
                db_path: str | Path = ENROLLED_DB_PATH) -> None:
    """이미지 파일에서 첫 번째 얼굴 encoding을 추출해 DB에 저장.

    같은 이름이 이미 있으면 덮어씀. 얼굴 미검출 시 ValueError.
    """
    image_path = Path(image_path)
    image_rgb = face_recognition.load_image_file(image_path)
    enroll_face_from_array(name, image_rgb, db_path)


def enroll_face_from_array(name: str, image_rgb: np.ndarray,
                            db_path: str | Path = ENROLLED_DB_PATH) -> None:
    """이미 로드된 RGB ndarray로 등록 (GUI 등에서 사용)."""
    db_path = Path(db_path)
    encodings = face_recognition.face_encodings(image_rgb)
    if not encodings:
        raise ValueError(f"얼굴을 찾을 수 없습니다: {name}")
    # 첫 번째 얼굴만 사용
    db = _load_db(db_path)
    db[name] = encodings[0]
    _save_db(db, db_path)


def load_enrolled(db_path: str | Path = ENROLLED_DB_PATH) -> dict[str, np.ndarray]:
    """DB 로드. 파일 없으면 빈 dict 반환."""
    return _load_db(Path(db_path))


def list_enrolled(db_path: str | Path = ENROLLED_DB_PATH) -> list[str]:
    """등록된 이름 목록."""
    return list(_load_db(Path(db_path)).keys())


def remove_enrolled(name: str, db_path: str | Path = ENROLLED_DB_PATH) -> bool:
    """등록 해제. 있었으면 True."""
    db_path = Path(db_path)
    db = _load_db(db_path)
    if name not in db:
        return False
    del db[name]
    _save_db(db, db_path)
    return True
