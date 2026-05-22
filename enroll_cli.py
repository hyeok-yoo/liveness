"""얼굴 등록 CLI.

사용:
  python enroll_cli.py --name 김철수                # 웹캠 (스페이스=캡처, q=취소)
  python enroll_cli.py --name 김철수 --image a.jpg  # 이미지 파일로 등록
  python enroll_cli.py --list                       # 등록자 목록
  python enroll_cli.py --remove 김철수              # 등록 해제
"""
from __future__ import annotations

import argparse
import sys

import cv2

from face_auth import enroll_face, enroll_face_from_array, list_enrolled, remove_enrolled


def _capture_from_webcam() -> "cv2.Mat | None":
    """웹캠 미리보기 → 스페이스 누르면 현재 프레임 반환, q면 None."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] 웹캠을 열 수 없습니다.", file=sys.stderr)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[capture] 정면을 보고 [Space] 키를 누르세요 (q=취소)")
    frame_bgr = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            preview = frame.copy()
            cv2.putText(preview, "[Space] Capture  [q] Cancel",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.imshow("Enroll - press Space to capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                frame_bgr = frame
                break
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return frame_bgr


def main() -> int:
    parser = argparse.ArgumentParser(description="Face enrollment CLI")
    parser.add_argument("--name", type=str, help="등록할 이름")
    parser.add_argument("--image", type=str, default=None,
                        help="이미지 파일 경로 (생략 시 웹캠 사용)")
    parser.add_argument("--list", action="store_true", help="등록자 목록 보기")
    parser.add_argument("--remove", type=str, default=None,
                        help="등록 해제할 이름")
    args = parser.parse_args()

    if args.list:
        names = list_enrolled()
        if not names:
            print("등록된 사용자 없음")
        else:
            print("등록자:")
            for n in names:
                print(f"  - {n}")
        return 0

    if args.remove:
        if remove_enrolled(args.remove):
            print(f"[OK] '{args.remove}' 등록 해제됨")
        else:
            print(f"[warn] '{args.remove}' 는 등록되어 있지 않음")
        return 0

    if not args.name:
        parser.error("--name 또는 --list/--remove 중 하나 필요")

    try:
        if args.image:
            enroll_face(args.name, args.image)
            print(f"[OK] '{args.name}' 등록 완료 (from {args.image})")
        else:
            frame = _capture_from_webcam()
            if frame is None:
                print("[abort] 캡처 취소됨")
                return 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enroll_face_from_array(args.name, rgb)
            print(f"[OK] '{args.name}' 등록 완료 (from webcam)")
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
