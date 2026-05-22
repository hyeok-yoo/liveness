"""평가 스크립트.

`data/{live,photo,screen}/` 아래 영상들을 일괄 처리해 FAR/FRR/ACER 계산.
또한 ablation 표를 출력해 보고서/발표 자료로 사용 가능.

사용:
  python evaluate.py                          # data/ 사용 + 기본 ablation
  python evaluate.py --data my_data/
  python evaluate.py --algorithm chrom        # rPPG 알고리즘 바꿔서 평가
  python evaluate.py --ablation               # 풀 ablation 표

영상 확장자: .mp4, .avi, .mov, .mkv
라벨 매핑:
  data/live/*    → live (정상)
  data/photo/*   → spoof (사진)
  data/screen/*  → spoof (영상 재생)
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from liveness_pipeline import LivenessPipeline, LivenessResult, _read_video_frames


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class Sample:
    path: Path
    label: int  # 1 = live, 0 = spoof
    category: str  # "live" | "photo" | "screen"


@dataclass
class Metrics:
    far: float          # False Accept Rate — spoof를 live로 잘못 본 비율 (낮을수록 좋음)
    frr: float          # False Reject Rate — live를 spoof로 잘못 본 비율 (낮을수록 좋음)
    acer: float         # (FAR + FRR) / 2
    accuracy: float
    n_live: int
    n_spoof: int
    confusion: dict     # {"tp": ..., "fp": ..., "tn": ..., "fn": ...}


def collect_samples(data_root: Path) -> list[Sample]:
    """data_root/{live,photo,screen}/*.mp4 등 수집."""
    samples: list[Sample] = []
    mapping = {"live": 1, "photo": 0, "screen": 0}
    for category, label in mapping.items():
        subdir = data_root / category
        if not subdir.exists():
            print(f"[warn] {subdir} 없음 — 건너뜀")
            continue
        for p in sorted(subdir.iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                samples.append(Sample(path=p, label=label, category=category))
    return samples


def evaluate_pipeline(
    pipeline: LivenessPipeline,
    samples: list[Sample],
    verbose: bool = True,
) -> tuple[Metrics, list[dict]]:
    """파이프라인을 모든 샘플에 적용 → Metrics + 개별 결과."""
    tp = fp = tn = fn = 0
    rows: list[dict] = []

    n_live = sum(1 for s in samples if s.label == 1)
    n_spoof = len(samples) - n_live

    for i, sample in enumerate(samples, 1):
        frames = _read_video_frames(str(sample.path))
        if not frames:
            if verbose:
                print(f"  [{i}/{len(samples)}] {sample.path.name}: 빈 비디오 — 스킵")
            continue

        t0 = time.time()
        res: LivenessResult = pipeline.analyze_frames(frames)
        dt = time.time() - t0

        pred = 1 if res.is_live else 0
        if sample.label == 1 and pred == 1:
            tp += 1
            outcome = "TP"
        elif sample.label == 0 and pred == 0:
            tn += 1
            outcome = "TN"
        elif sample.label == 1 and pred == 0:
            fn += 1
            outcome = "FN"
        else:
            fp += 1
            outcome = "FP"

        rows.append({
            "name": sample.path.name,
            "category": sample.category,
            "true": "live" if sample.label == 1 else "spoof",
            "pred": "live" if pred == 1 else "spoof",
            "outcome": outcome,
            "reasons": "; ".join(res.reasons[:3]),
            "time_s": round(dt, 2),
        })

        if verbose:
            print(f"  [{i}/{len(samples)}] {sample.path.name} "
                  f"({sample.category}) → {'LIVE' if pred else 'SPOOF'} "
                  f"[{outcome}] ({dt:.1f}s)")

    # 메트릭
    far = fp / n_spoof if n_spoof else 0.0
    frr = fn / n_live if n_live else 0.0
    acer = (far + frr) / 2.0
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0

    metrics = Metrics(
        far=far, frr=frr, acer=acer, accuracy=accuracy,
        n_live=n_live, n_spoof=n_spoof,
        confusion={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    )
    return metrics, rows


def print_metrics(label: str, m: Metrics) -> None:
    print(f"\n--- {label} ---")
    print(f"  N_live={m.n_live}, N_spoof={m.n_spoof}")
    print(f"  Confusion: TP={m.confusion['tp']}, FP={m.confusion['fp']}, "
          f"TN={m.confusion['tn']}, FN={m.confusion['fn']}")
    print(f"  FAR (false accept): {m.far*100:5.1f}%")
    print(f"  FRR (false reject): {m.frr*100:5.1f}%")
    print(f"  ACER:               {m.acer*100:5.1f}%")
    print(f"  Accuracy:           {m.accuracy*100:5.1f}%")


def run_ablation(samples: list[Sample], fs: float) -> None:
    """여러 구성으로 평가해 비교 표 출력. 발표/보고서 자료."""
    configs = [
        ("Pre+Moire+rPPG(POS) [전체]",
            dict(rppg_algorithm="pos", enable_pre_screen=True,
                 enable_moire_fft=True, enable_rppg=True)),
        ("Pre+Moire+rPPG(CHROM)",
            dict(rppg_algorithm="chrom", enable_pre_screen=True,
                 enable_moire_fft=True, enable_rppg=True)),
        ("Moire only (FFT)",
            dict(enable_pre_screen=False, enable_moire_fft=True, enable_rppg=False)),
        ("rPPG(POS) only",
            dict(rppg_algorithm="pos", enable_pre_screen=False,
                 enable_moire_fft=False, enable_rppg=True)),
        ("rPPG(CHROM) only",
            dict(rppg_algorithm="chrom", enable_pre_screen=False,
                 enable_moire_fft=False, enable_rppg=True)),
    ]

    print("\n" + "=" * 78)
    print(f"{'Configuration':<35} {'FAR%':>8} {'FRR%':>8} {'ACER%':>8} {'Acc%':>8}")
    print("-" * 78)

    for name, kwargs in configs:
        pipeline = LivenessPipeline(fs=fs, **kwargs)
        m, _ = evaluate_pipeline(pipeline, samples, verbose=False)
        print(f"{name:<35} {m.far*100:>7.1f}  {m.frr*100:>7.1f}  "
              f"{m.acer*100:>7.1f}  {m.accuracy*100:>7.1f}")
        pipeline.close()
    print("=" * 78)


def write_csv(rows: list[dict], path: Path) -> None:
    """샘플별 결과 CSV 저장."""
    if not rows:
        return
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {path} 저장")


def main() -> int:
    parser = argparse.ArgumentParser(description="Liveness evaluation")
    parser.add_argument("--data", type=str, default="data",
                        help="데이터 루트 (live/, photo/, screen/ 하위)")
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--algorithm", choices=("pos", "chrom"), default="pos")
    parser.add_argument("--ablation", action="store_true",
                        help="여러 구성을 비교하는 표 출력")
    parser.add_argument("--csv", type=str, default=None,
                        help="샘플별 결과 CSV 출력 경로")
    args = parser.parse_args()

    data_root = Path(args.data)
    samples = collect_samples(data_root)
    if not samples:
        print(f"[error] {data_root} 에서 영상을 찾지 못함. data/live/, data/photo/, "
              f"data/screen/ 에 영상을 넣어주세요.", file=sys.stderr)
        return 1

    print(f"[load] {len(samples)} 샘플 수집 — "
          f"live={sum(1 for s in samples if s.label==1)}, "
          f"spoof={sum(1 for s in samples if s.label==0)}")

    if args.ablation:
        run_ablation(samples, fs=args.fs)
        return 0

    pipeline = LivenessPipeline(fs=args.fs, rppg_algorithm=args.algorithm)
    metrics, rows = evaluate_pipeline(pipeline, samples, verbose=True)
    pipeline.close()

    print_metrics(f"전체 ({args.algorithm})", metrics)

    if args.csv:
        write_csv(rows, Path(args.csv))

    return 0


if __name__ == "__main__":
    sys.exit(main())
