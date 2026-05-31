# 기술 문서 — 2D 카메라 기반 Face Liveness Detection (rPPG + Moiré)

> 본 문서는 보고서/포트폴리오용 기술 설명서입니다. 모든 수치와 동작은 실제 소스 코드(`liveness_pipeline.py`, `rppg/`, `moire/`, `face_detector.py`, `face_auth/`, `evaluate.py`)에 근거합니다.

---

## 1. 시스템 개요 & 아키텍처

### 1.1 목표

일반 2D RGB 웹캠 한 대만으로, 등록된 사용자가 **실제 살아있는 사람**일 때만 인증을 통과시키고, **사진(인쇄/디스플레이) 및 영상 재생(replay) 공격**은 차단하는 anti-spoofing PoC.

특수 하드웨어(IR/depth 카메라)가 없는 환경을 가정하므로, 깊이 정보 대신 **생체 신호(rPPG, 심박)**와 **재촬영 아티팩트(Moiré 패턴)**를 단서로 사용한다.

### 1.2 다층 방어(multi-layer defense) 파이프라인

판정은 `liveness_pipeline.py`의 `LivenessPipeline.analyze_frames()`가 오케스트레이션한다. 각 계층은 서로 다른 공격 유형을 차단하므로 single point of failure가 없다. **OR 로직**: 어느 한 계층이라도 spoof 신호를 내면 즉시 거부(보안 측면에서 보수적, early-exit).

```
                 [ 입력: BGR 프레임 시퀀스 (약 10초, ~30fps) ]
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │ 0) 프레임 수 < 2 → 거부 (not_enough_frames)          │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 1) Pre-screen  (pre_screen.py)                      │
        │    조명 변화 std / Lucas-Kanade optical flow 모션    │
        │    → 임계값 초과 시 거부 (early-exit)                │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 2) 얼굴 검출 + RGB 시계열 추출 (단일 패스)           │
        │    MediaPipe FaceLandmarker(468 landmark)            │
        │    ROI별 skin-pixel mean RGB + 대표 프레임 보관      │
        │    얼굴 검출 프레임 < min_face_frames(30) → 거부     │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 3) Moiré FFT  (moire/fft_detector.py)               │
        │    얼굴 crop의 mid-high 주파수 링 에너지 비율        │
        │    score > 0.18 → 거부 (영상 replay 차단)           │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 4) Moiré LBP+SVM (선택)  (moire/lbp_svm_detector.py)│
        │    학습된 모델이 있을 때만 동작 (기본 비활성)        │
        │    spoof 확률 ≥ 0.5 → 거부                          │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 5) rPPG spoof check  (rppg/spoof_check.py)          │
        │    POS/CHROM 맥박 추출 + background-relative 판정    │
        │    decisive 모드: spoof → 거부                      │
        │    demo 모드: spoof → 경고만 (거부권 없음)          │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 6) Face Recognition  (face_auth/recognize.py)       │
        │    LIVE 통과 후 대표 프레임으로 dlib 매칭           │
        │    distance < 0.6 → 환영 / 아니면 거부              │
        └─────────────────────────────────────────────────────┘
```

> 6) Face Recognition은 liveness와 분리되어 `auth_pipeline.py`의 `AuthPipeline.authenticate_frames()`에서 호출된다. liveness가 통과해야만 비로소 신원 매칭으로 넘어간다.

### 1.3 모듈 맵

| 파일 | 역할 |
|---|---|
| `liveness_pipeline.py` | liveness 계층 통합 (`LivenessPipeline`) |
| `auth_pipeline.py` | liveness + 얼굴인식 통합 (`AuthPipeline`) |
| `pre_screen.py` | 조명/모션 사전 거부 (`PreScreener`) |
| `face_detector.py` | MediaPipe ROI/마스크/skin mask (`FaceROIDetector`) |
| `rppg/extractor.py` | ROI별 RGB 시계열 누적 (`RGBTimeSeriesExtractor`) |
| `rppg/pos.py`, `rppg/chrom.py` | 맥박 추출 알고리즘 |
| `rppg/signal_processing.py` | detrend / FFT / BPM / SNR |
| `rppg/spoof_check.py` | rPPG 기반 spoof 판정 (`RPPGSpoofChecker`) |
| `moire/fft_detector.py` | FFT 기반 Moiré 검출 (룰 기반) |
| `moire/lbp_svm_detector.py` | LBP+SVM Moiré 검출 (학습형) |
| `face_auth/enroll.py`, `face_auth/recognize.py` | 얼굴 등록/인식 |
| `evaluate.py` | FAR/FRR/ACER, ablation 평가 |

---

## 2. 각 방어 계층 상세

### 2.1 Pre-screen (`pre_screen.py`의 `PreScreener.analyze()`)

**목적**: rPPG 분석 전에 환경 노이즈를 사전 거부. 조명 급변이나 큰 모션은 rPPG에 **가짜 맥박(false pulse)**을 만들어 내므로, 그 전에 끊어 준다.

**입력**: BGR 프레임 시퀀스. **출력**: `PreScreenResult(accepted, illumination_std, motion_magnitude, reasons, korean_reasons)`.

**원리 / 의사코드**:

```
grays = [grayscale(f) for f in frames]            # max_frames(30) stride 샘플링

# (1) 조명 변화: 프레임 평균 밝기 시계열의 표준편차
illum_std = std([ mean(g) for g in grays ])

# (2) 모션: 인접 프레임 쌍에 Lucas-Kanade sparse optical flow
for (prev, next) in zip(grays[:-1], grays[1:]):
    pts = goodFeaturesToTrack(prev)               # 코너 특징점
    next_pts, status = calcOpticalFlowPyrLK(prev, next, pts)
    motion += mean(|next_pts - pts|)              # 추적 성공 점들의 평균 이동량(px)
motion_magnitude = mean(motion)

reject if illum_std > 40.0  OR  motion_magnitude > 25.0
```

**임계값과 근거**:

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `illumination_std_threshold` | 40.0 | 밝기 std 상한. 초과 = 조명 깜빡임/급변 |
| `motion_threshold` | 25.0 | LK 평균 이동량(px/frame) 상한. 초과 = 카메라/피사체 큰 움직임 |
| `max_frames` | 30 | stride 샘플링으로 분석량을 30프레임으로 제한 (메모리/속도) |

LK 파라미터는 `winSize=(15,15)`, `maxLevel=2`(피라미드 2단계), `maxCorners=80`, `qualityLevel=0.01`. 특징점은 첫 프레임에서 매 쌍마다 새로 추출 — PoC 규모에 충분한 단순화.

> 부수 효과: 사진을 **천천히 흔드는** 공격은 모션이 임계값을 넘어 pre-screen에서 거부되거나, 얼굴 검출 프레임 부족으로 다음 단계(`too_few_face_frames`)에서 막힌다.

### 2.2 얼굴 검출 & ROI 추출 (`face_detector.py`)

**MediaPipe Tasks API의 `FaceLandmarker`** (IMAGE 모드, `num_faces=1`)로 프레임당 **468개 landmark**를 얻는다. legacy `face_mesh`가 0.10.15+에서 제거되어 Tasks API를 사용하며, 모델(`models/face_landmarker.task`)은 첫 실행 시 자동 다운로드한다.

landmark 인덱스로 **convex hull 마스크**를 만들어 ROI를 정의한다:

| ROI | 용도 |
|---|---|
| `forehead`, `left_cheek`, `right_cheek` | rPPG 맥박 추출에 사용되는 3개 얼굴 ROI |
| `nose` | ROI 마스크는 생성하나 spoof 판정에는 미사용 |
| `background` | 얼굴 bbox(+10% margin) **바깥** 영역 → 상대 판정 기준 |

**Skin-pixel masking** (`mean_rgb_in_mask(..., skin_only=True)` + `_skin_mask()`): YCrCb 색공간에서 **Cr ∈ [133,173], Cb ∈ [77,127]** (Garcia & Tziritas 1999) 범위만 피부로 간주해, ROI 안의 머리카락/안경/그림자를 제외하고 피부 픽셀만 평균낸다. 이는 PPG SNR을 크게 높인다(ROBUSTNESS_CHANGES 기준 +2~3 dB). 단, 피부 픽셀이 원래 마스크의 30% 미만이면(측면 얼굴/검출 실패) 원래 마스크로 **자동 폴백**한다. 배경 ROI는 피부가 없으므로 `skin_only=False`.

`get_face_crop()`은 bbox로 얼굴을 잘라 Moiré 검출에 넘긴다. 얼굴이 처음 검출된 프레임은 face_recognition용 `representative_frame`으로 보관한다.

### 2.3 RGB 시계열 추출 (`rppg/extractor.py`)

`RGBTimeSeriesExtractor`는 프레임마다 `process_frame_with_detection()`으로 검출과 ROI 평균 RGB를 **한 번의 패스**로 누적한다(중복 detect 호출 방지). 얼굴 미검출 프레임은 NaN 행으로 채우고, `get_signals(interpolate=True)`에서 **채널별 선형 보간**으로 짧은 결손 구간을 메운다(`np.interp`). 결과는 `{roi_name: (T,3)}` 형태.

### 2.4 Moiré FFT (`moire/fft_detector.py`의 `MoireFFTDetector`)

**목적**: 폰/모니터 화면을 카메라로 **재촬영**할 때 카메라 센서 격자와 디스플레이 픽셀 격자가 간섭하며 생기는 Moiré(줄무늬) → mid-high 주파수 대역 에너지 증가. **학습 데이터 없이 임계값만으로** 동작.

**입력**: 얼굴 crop 리스트. **출력**: `MoireFFTResult(is_spoof, score, threshold)`.

**알고리즘** (`_compute_score`):

```
1. grayscale → 256×256 resize (patch_size)
2. 평균 제거 (DC가 스펙트럼을 지배하는 것 방지)
3. 2D Hann 윈도우 곱 (경계 불연속 → spurious 고주파 억제)
4. 2D FFT → |F| → fftshift → log(1+|F|)   (동적 범위 압축)
5. score = (링 마스크 영역 에너지 합) / (전체 에너지 합)
```

링 마스크: 중심 기준 정규화 반경 **r ∈ [0.25, 0.45]** (low_band~high_band)의 annular 영역. `analyze_video()`는 여러 프레임 score를 **평균**해 단일 프레임보다 안정화한다.

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `patch_size` | 256 | FFT 입력 크기(2의 거듭제곱) |
| `low_band` / `high_band` | 0.25 / 0.45 | mid-high 링의 내경/외경(정규화 주파수) |
| `threshold` | **0.18** | score가 이 값 초과면 spoof |

> **근거와 한계**: 0.18은 환경에 민감한 경험적 값. 카메라/화면/거리 조합에 따라 적정값이 달라지므로, 실데이터 수집 후 LBP+SVM(2.5)으로 보강하는 것이 향후 과제이다.

### 2.5 Moiré LBP+SVM (`moire/lbp_svm_detector.py`의 `MoireLBPSVMDetector`)

**목적**: FFT 임계값의 환경 민감성을 보완하는 **학습형** 텍스처 분류기. Moiré는 규칙적 고주파 텍스처를 형성하므로 LBP가 잘 포착한다. **기본 비활성**(`enable_moire_lbp=False`)이며 학습된 모델이 있을 때만 동작.

- **특징**: grayscale 128×128 → uniform LBP(`P=8, R=1`) 히스토그램 (bin 수 = P+2 = 10), `density=True`로 정규화.
- **분류기**: `StandardScaler → SVC(RBF, probability=True, class_weight="balanced")` 파이프라인. 소규모 데이터(클래스당 ~10장)에서도 과적합을 억제.
- **추론**: `predict_proba`로 spoof 확률 산출, `classes_`에서 라벨 1의 인덱스를 명시적으로 찾아 사용. **확률 ≥ 0.5 → spoof**. `fit()` 전 호출 시 `NotFittedError`.
- 파이프라인에서는 추론 비용을 줄이려 **중간 프레임 하나**(`face_crops[len//2]`)만 분석한다.

### 2.6 rPPG (`rppg/`)

#### 2.6.1 맥박 추출

ROI별 평균 RGB 시계열(T×3)을 입력으로 펄스 1차원 신호를 만든다. `algorithm` 파라미터로 둘 중 선택.

- **POS** (`pos.py`, Wang et al. 2017): 슬라이딩 윈도우(`window_sec=1.6`)마다 RGB를 평균으로 정규화한 뒤, 투영 행렬 `P=[[0,1,-1],[-2,1,1]]`로 두 축 신호 S0, S1을 만들고, `alpha = std(S0)/std(S1)`로 결합(`h = S0 + alpha·S1`)해 **overlap-add**로 합친다.
- **CHROM** (`chrom.py`, de Haan & Jeanne 2013): 정규화 RGB에서 색차 신호 `Xs = 3R−2G`, `Ys = 1.5R+G−1.5B`를 만들고 각각 0.7~4.0 Hz 대역통과 후 `alpha = std(Xs)/std(Ys)`로 결합(`Xs − alpha·Ys`).

#### 2.6.2 신호 처리 (`rppg/signal_processing.py`)

10초 같은 **짧은 캡처**에서 안정적인 BPM/SNR을 얻기 위한 4가지 핵심 기법:

| 기법 | 함수 | 왜 중요한가 (짧은 캡처에서) |
|---|---|---|
| **Smooth-priors detrending** (Tarvainen 2002, λ=100) | `detrend()` | 2차 차분 행렬 정규화로 저주파 베이스라인 드리프트를 강력 제거. 안 하면 DC 드리프트가 0.7 Hz 근방으로 누설되어 **가짜 peak** 발생 |
| **Hanning window + 4× zero-padding FFT** | `_power_spectrum()` | 윈도우로 **spectral leakage** 억제, zero-padding(nfft=max(N×4,1024))으로 BPM **분해능** 향상(약 6→1.5 BPM). 짧은 신호는 분해능이 낮아 peak가 흐려짐 |
| **Harmonic-aware SNR** (1차+2차 harmonic) | `compute_snr_db()` | 실제 PPG는 1·2차 harmonic에 에너지가 **분산**되므로 둘 다 신호로 인정 → SNR +2~3 dB |
| **Parabolic peak interpolation** | `_parabolic_interp()` | FFT bin 사이를 3점 포물선 보간해 이산 격자보다 정밀한 BPM 추정 |

`compute_snr_db`는 fundamental ±(band/2=6 BPM) 및 2차 harmonic 대역을 signal로, valid_range(42~180 BPM, spoof_check에서는 42~150) 내 나머지를 noise로 정의하고 **10·log10(S/N)** dB를 반환한다. `bandpass()`는 zero-phase `filtfilt` Butterworth(order 4).

#### 2.6.3 spoof 판정 — 다음 절(3장)에서 상세히 다룬다.

### 2.7 Face Recognition (`face_auth/`)

liveness 통과 후에만 동작. **`face_recognition`/dlib**으로 대표 프레임에서 첫 얼굴 위치를 찾고 128-d encoding을 추출, 등록 DB의 encoding들과 **euclidean distance**를 비교한다.

- 등록(`enroll.py`): 이미지에서 첫 얼굴 encoding을 추출해 `data/enrolled/db.pkl`에 `{name: encoding}`으로 저장(같은 이름은 덮어씀).
- 인식(`recognize.py`): 최근접 distance가 **threshold(0.6)** 미만이면 매칭. 등록자 0명이면 에러 없이 `matched=False`.
- 통합(`auth_pipeline.py`): `match` / `no_match` / `no_face` / `spoof_blocked` 4가지 `stage`로 GUI 메시지를 결정.

---

## 3. rPPG 판정 로직 (이 프로젝트의 핵심 차별점)

전부 `rppg/spoof_check.py`의 `RPPGSpoofChecker.analyze()`에 구현. 핵심 ROI는 `_FACE_ROIS = (forehead, left_cheek, right_cheek)`와 `background`.

### 3.1 결정적 아이디어 — Background-relative margin

**문제**: 절대 SNR은 조명/카메라/거리에 따라 매번 달라진다. 절대 임계값으로 판정하면 실제 얼굴이 거부(false reject)되거나, 임계값을 낮추면 사진이 통과(false accept)된다.

**통찰**: rPPG의 본질은 "**얼굴 피부에만 맥박이 존재하고 배경에는 없다**"는 것이다. 그렇다면 환경 변동에 강건한 신호는 절대값이 아니라 **얼굴 SNR과 배경 SNR의 차이(margin)**다.

```
사진:    face_snr ≈ bg_snr        (둘 다 환경 노이즈)        → margin ≈ 0 이하
영상:    face_snr ≈ bg_snr        (배경의 픽셀 그리드 노이즈) → margin 작음
실제얼굴: face_snr ≫ bg_snr        (얼굴에만 pulse)            → margin 큰 양수
```

따라서 `snr_margin = fused_snr − bg_snr`가 핵심 판정 신호다. 배경 SNR을 신뢰할 수 없으면(sentinel) margin 검사는 `inf`로 건너뛴다.

### 3.2 보조 설계 4가지

1. **SNR-weighted multi-ROI fusion** (`_fused_metrics`): 3개 얼굴 ROI 펄스를 z-score 정규화한 뒤, 각 ROI의 SNR을 선형 비율(`10^(snr/10)`, −10 dB로 floor)로 가중해 평균. 강한 ROI에 무게를 더 줘서 한 ROI가 망가져도 fused SNR이 크게 떨어지지 않는다. 신호 융합으로 약 **√N배 SNR 향상**.

2. **Multi-window best-of-N** (`_best_window_metrics`): 10초 신호를 6초 윈도우 × stride 2초로 슬라이딩(약 3개) + 전체(1개) = **4개 후보** 중 **best SNR**을 채택. 캡처 중 잠깐 흔들리거나 깜빡여도 양호한 구간만으로 평가.

3. **Best-ROI fallback**: 한 ROI라도 배경보다 충분히 강하면 통과(아래 조건 c). 한쪽 뺨이 그림자/머리카락에 가려져도 OK.

4. **Adaptive margin**: 배경이 매우 조용한 환경(`bg_snr < −8 dB`)에서는 절대 노이즈 자체가 낮으므로 요구 margin을 1.0 dB로 낮춘다(사진은 face≈bg라 여전히 안전).

### 3.3 통과 / 거부 조건

펄스는 0.7~2.5 Hz(42~150 BPM, 정상 심박)로 좁게 대역통과한다.

**약한 맥박(weak_pulse) 검사** — 아래 셋 중 **하나라도** 만족하면 통과:

| 조건 | 코드 | 의미 |
|---|---|---|
| (a) margin pass | `snr_margin ≥ effective_margin` | 얼굴 fused가 배경보다 충분히 강함 (**핵심 신호**) |
| (b) absolute pass | `fused_snr ≥ snr_threshold_db(3.0)` AND `fused_snr > bg_snr` | 절대 신호가 강함 |
| (c) best-ROI pass | `best_roi_margin ≥ effective_margin + 1.0` | 한 ROI라도 배경보다 분명히 강함 |

셋 다 실패하면 `weak_pulse` 사유로 spoof.

**추가 정합성 검사 두 가지** (위반 시 spoof):

- **BPM ROI 일관성**: 배경 대비 일정 margin 이상 SNR을 가진 "신뢰 ROI"가 2개 이상일 때, 그들 BPM의 spread가 `bpm_agreement(20)` BPM을 넘으면 거부. 실제 맥박이라면 부위별 BPM이 일치해야 한다.
- **배경 주파수 매칭**: 배경 SNR이 임계값 이상이고 margin이 부족할 때, 배경 BPM이 fused BPM과 `bg_match_bpm(5)` BPM 이내로 일치하면 거부. 얼굴과 배경에 같은 주파수 = 환경 노이즈.

### 3.4 한글 거부 사유 (UI용)

각 거부 조건은 사용자에게 보일 자연어 `korean_reasons`를 함께 채운다(`reasons`와 1:1). 예:

```
심박 신호 미감지 — 얼굴 맥박이 배경 노이즈와 구분되지 않음
  (수치: 얼굴-배경 SNR -2.6dB < 기준 +1.0dB)
```

원시 에러 코드(`rppg:weak_pulse: ...`) 대신 위와 같이 수치를 포함한 설명을 보여 줘 발표/시연 시 가독성을 높였다. `auth_pipeline.py`는 한글 사유를 우선 사용하고, 없으면 원시 코드로 폴백한다.

### 3.5 유지보수성 리팩토링 — sentinel 상수

이전에는 `-999.0`(분석 불가)·`-100.0`(유효 SNR 하한) 같은 매직 넘버가 코드 곳곳에 흩어져 있었다. 이를 모듈 상수 **`_SNR_INVALID = -999.0`**, **`_SNR_FLOOR = -100.0`**로 추출해 의미를 명시하고, "유효한 SNR인가"는 `snr > _SNR_FLOOR`로 일관 판정하도록 정리했다.

### 3.6 decisive vs demo 모드 (`liveness_pipeline.py`)

- **decisive 모드(`rppg_decisive=True`, 기본)**: rPPG가 spoof로 판정하면 즉시 거부.
- **demo 모드(`rppg_decisive=False`)**: rPPG가 spoof여도 **거부권 없이 경고(`[경고] ...`)만** 표시하고 LIVE로 통과. 발표장처럼 카메라/조명/거리가 미지수인 환경에서 본인이 false reject되는 것을 막기 위함. **사진·영상 위협은 이미 pre-screen + Moiré가 막고 있으므로** 보안은 유지된다.

---

## 4. 수치 설정값 표 (threshold 모음)

| 계층 | 파라미터 | 기본값 | 정의 위치 | 의미 |
|---|---|---|---|---|
| 파이프라인 | `min_face_frames` | 30 | `LivenessPipeline` | 최소 얼굴 검출 프레임 |
| Pre-screen | `prescreen_illumination_std` | 40.0 | `PreScreener` | 밝기 std 상한 |
| Pre-screen | `prescreen_motion` | 25.0 | `PreScreener` | LK 모션 상한(px/frame) |
| Moiré FFT | `moire_fft_threshold` | 0.18 | `MoireFFTDetector` | 링 에너지 비율 상한 |
| Moiré FFT | `low_band`/`high_band` | 0.25 / 0.45 | `MoireFFTDetector` | 링 내경/외경(정규화 주파수) |
| Moiré LBP | spoof 확률 컷 | 0.5 | `MoireLBPSVMDetector` | SVM spoof 확률 임계 |
| rPPG | `snr_threshold_db` | 3.0 | `RPPGSpoofChecker` | 절대 SNR 통과 임계(fallback) |
| rPPG | `snr_margin_db` | 1.0 | `RPPGSpoofChecker` | 얼굴−배경 SNR margin (핵심) |
| rPPG | `bpm_agreement` | 20.0 | `RPPGSpoofChecker` | 신뢰 ROI BPM spread 상한 |
| rPPG | `bg_match_bpm` | 5.0 | `RPPGSpoofChecker` | 배경-얼굴 BPM 일치 판정폭 |
| rPPG | adaptive margin trigger | bg_snr < −8.0 | `analyze()` | 조용한 환경 margin 완화 |
| rPPG | 펄스 대역통과 | 0.7~2.5 Hz | `analyze()` | 42~150 BPM |
| rPPG | multi-window | 6초/stride 2초 | `_best_window_metrics` | best-of-N 윈도우 |
| 신호처리 | detrend λ | 100.0 | `detrend()` | smooth-priors 강도 |
| 신호처리 | zero-pad factor | 4 (nfft≥1024) | `_power_spectrum` | FFT 분해능 |
| 신호처리 | SNR band width | 12.0 BPM | `compute_snr_db` | signal 대역폭 |
| 얼굴인식 | distance threshold | 0.6 | `FaceRecognizer` | dlib encoding 매칭 |
| skin mask | Cr/Cb 범위 | [133,173]/[77,127] | `_skin_mask` | YCrCb 피부 |
| skin mask | 폴백 임계 | 0.3 | `mean_rgb_in_mask` | 피부 픽셀 최소 비율 |

---

## 5. 평가 방법 (`evaluate.py`)

`data/{live,photo,screen}/`의 영상들을 일괄 처리한다. 라벨: `live → 1(live)`, `photo`·`screen → 0(spoof)`.

**지표** (`Metrics`):

- **FAR** (False Accept Rate) = FP / N_spoof — spoof를 live로 잘못 통과시킨 비율 (낮을수록 안전)
- **FRR** (False Reject Rate) = FN / N_live — 실제 얼굴을 거부한 비율 (낮을수록 편의성)
- **ACER** = (FAR + FRR) / 2 — 종합 오류율
- **Accuracy** = (TP + TN) / Total

**Ablation** (`run_ablation`, `--ablation`): 구성별로 FAR/FRR/ACER/Acc 비교 표를 출력해 각 계층의 기여도를 보인다.

| 구성 | 의미 |
|---|---|
| Pre+Moire+rPPG(POS) [전체] | 전체 파이프라인 (기본) |
| Pre+Moire+rPPG(CHROM) | rPPG 알고리즘만 CHROM으로 교체 |
| Moire only (FFT) | Moiré 단독 |
| rPPG(POS) only | rPPG 단독 |
| rPPG(CHROM) only | rPPG 단독(CHROM) |

`enable_*` 플래그가 `LivenessPipeline` 생성자에 노출되어 있어 계층을 켜고 끄며 ablation을 구성할 수 있다. `--csv`로 샘플별 결과(outcome, reasons, 처리시간)를 저장한다.

---

## 6. 한계와 향후 과제

1. **Moiré FFT 임계값(0.18)의 환경 민감성**: 카메라/디스플레이/거리 조합에 따라 적정값이 달라진다. 실데이터를 수집해 **LBP+SVM(2.5)을 학습**시켜 룰 기반 임계값을 보강하는 것이 1순위 과제.

2. **rPPG는 ML이 아닌 룰 기반**: POS/CHROM과 SNR/margin 판정은 모두 **물리 원리(혈류에 의한 피부 반사율 변화)에 기반한 규칙**이다. 장점은 (a) **학습 데이터 불필요**, (b) 각 거부 사유가 수치로 설명 가능. 단점은 임계값을 손으로 튜닝해야 한다는 점. ML 도입은 데이터가 충분히 쌓이고 설명가능성을 일부 포기할 수 있을 때 검토.

3. **10초 캡처 필요**: 짧은 신호에서 안정적 SNR을 얻기 위해 신호처리를 강화했지만, 그래도 수 초 이상의 캡처가 필요해 실시간성에 제약.

4. **측면/저조도 취약**: skin-pixel masking이 30% 폴백으로 보완하지만, 큰 측면각이나 저조도에서는 ROI 피부 픽셀이 부족해 SNR이 떨어진다.

5. **웹/앱 이식**: 현재는 데스크톱(tkinter GUI). 향후 브라우저(WebRTC)·모바일로 이식하려면 MediaPipe/dlib 의존성과 프레임 처리 비용을 경량화해야 한다.

---

### 참고문헌

- POS: Wang et al., "Algorithmic Principles of Remote PPG," IEEE TBME 2017
- CHROM: de Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG," IEEE TBME 2013
- Moiré: Patel, Han, Jain, "Use of Moiré Patterns to Detect Replay Video Attacks," ICB 2015
- Detrending: Tarvainen et al., "An advanced detrending method with application to HRV analysis," IEEE TBME 2002
- Skin: Garcia & Tziritas, "Face detection using quantized skin color regions merging," 1999
