"""CLI 데모.

웹캠 또는 영상 파일로 liveness/auth 검사를 빠르게 시험.

사용:
  python demo_cli.py                      # 웹캠 15초 캡처 후 liveness 검사
  python demo_cli.py --video data/live/foo.mp4
  python demo_cli.py --auth               # 등록자 매칭까지 (face_recognition 필요)
  python demo_cli.py --algorithm chrom    # CHROM 알고리즘 사용
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from liveness_pipeline import LivenessPipeline, LivenessResult, _read_video_frames


def capture_from_webcam(duration_sec: float = 15.0, fps_hint: float = 30.0) -> list:
    """웹캠에서 duration_sec 동안 프레임을 모아 list로 반환."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] 웹캠을 열 수 없습니다.", file=sys.stderr)
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps_hint)

    frames: list = []
    start = time.time()
    print(f"[capture] {duration_sec}초 캡처 시작 — 카메라를 정면으로 응시하세요...")

    while time.time() - start < duration_sec:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        # 진행률 출력 (1초마다)
        elapsed = time.time() - start
        remaining = duration_sec - elapsed
        if int(elapsed * 10) % 10 == 0:
            print(f"  남은 시간: {remaining:.1f}s  (frames={len(frames)})", end="\r")

    cap.release()
    print(f"\n[capture] 종료 — 총 {len(frames)}프레임 수집")
    return frames


def print_liveness(res: LivenessResult) -> None:
    """LivenessResult 사람이 읽기 좋게 출력."""
    print("=" * 60)
    print(f"  IS_LIVE = {res.is_live}")
    print(f"  frames: total={res.frames_total}, with_face={res.frames_with_face}")
    if res.reasons:
        print("  reasons:")
        for r in res.reasons:
            print(f"    - {r}")
    if res.pre_screen is not None:
        ps = res.pre_screen
        print(f"  pre_screen: illum_std={ps.illumination_std:.2f}, "
              f"motion={ps.motion_magnitude:.2f}, accepted={ps.accepted}")
    if res.moire_fft is not None:
        m = res.moire_fft
        print(f"  moire_fft: score={m.score:.4f} (thr={m.threshold}) "
              f"spoof={m.is_spoof}")
    if res.rppg is not None:
        r = res.rppg
        print("  rppg:")
        for roi, bpm in r.bpms.items():
            print(f"    {roi}: bpm={bpm:.1f}, snr={r.snrs[roi]:.1f} dB")
        print(f"    background: bpm={r.background_bpm:.1f}, snr={r.background_snr_db:.1f} dB")
        margin = r.fused_snr_db - r.background_snr_db if r.background_snr_db > -100 else float('nan')
        print(f"    >>> FUSED: bpm={r.fused_bpm:.1f}, snr={r.fused_snr_db:.1f} dB, "
              f"margin(face-bg)={margin:+.1f} dB")
        print(f"    is_spoof={r.is_spoof}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Face liveness CLI demo")
    parser.add_argument("--video", type=str, help="영상 파일 경로 (없으면 웹캠 사용)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="웹캠 캡처 시간(초)")
    parser.add_argument("--fs", type=float, default=30.0, help="샘플링 주파수 (fps)")
    parser.add_argument("--algorithm", choices=("pos", "chrom"), default="pos",
                        help="rPPG 알고리즘")
    parser.add_argument("--auth", action="store_true",
                        help="등록자 매칭까지 수행 (face_recognition 필요)")
    parser.add_argument("--no-prescreen", action="store_true")
    parser.add_argument("--no-moire", action="store_true")
    parser.add_argument("--no-rppg", action="store_true")
    parser.add_argument(
        "--demo", action="store_true",
        help="발표/시연용 soft 모드: rPPG 약하면 경고만 표시하고 LIVE 통과 "
             "(pre-screen+moire가 사진/영상 공격을 막음)",
    )
    args = parser.parse_args()

    # 프레임 수집
    if args.video:
        path = Path(args.video)
        if not path.exists():
            print(f"[error] 파일 없음: {path}", file=sys.stderr)
            return 1
        print(f"[load] {path}")
        frames = _read_video_frames(str(path))
        print(f"[load] {len(frames)} 프레임 로드됨")
    else:
        frames = capture_from_webcam(duration_sec=args.duration, fps_hint=args.fs)

    if not frames:
        print("[error] 프레임 없음 — 종료")
        return 1

    # 파이프라인 실행
    pipeline = LivenessPipeline(
        fs=args.fs,
        rppg_algorithm=args.algorithm,
        enable_pre_screen=not args.no_prescreen,
        enable_moire_fft=not args.no_moire,
        enable_rppg=not args.no_rppg,
        rppg_decisive=not args.demo,
    )

    if args.auth:
        # face_recognition 사용 — 미설치 시 에러
        from auth_pipeline import AuthPipeline
        from face_auth import FaceRecognizer
        auth = AuthPipeline(liveness=pipeline, recognizer=FaceRecognizer())
        auth_res = auth.authenticate_frames(frames)
        print_liveness(auth_res.liveness)
        print(f"\n[auth] stage={auth_res.stage}, success={auth_res.success}")
        print(f"  message: {auth_res.message}")
        auth.close()
    else:
        res = pipeline.analyze_frames(frames)
        print_liveness(res)
        pipeline.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
