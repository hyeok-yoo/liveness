# 2D 카메라 기반 Face Liveness Detection (rPPG + Moiré)

고등학교 졸업 프로젝트. 2D RGB 웹캠만 사용해 사진/영상 spoof 공격을 막는 face liveness detection PoC.

## 핵심 아이디어
- **rPPG (remote photoplethysmography)**: 얼굴 피부 색상의 미세 변화로 심박 검출 → 사진은 심박 없음
- **Moiré 패턴**: 폰/모니터 화면을 카메라로 재촬영할 때 발생하는 줄무늬 → 영상 replay 차단
- **얼굴인식**: liveness 통과 후 `face_recognition` 라이브러리로 등록자 매칭

## 시스템 구조
```
카메라 → Face Detection (MediaPipe)
       → Pre-screening (조명/모션)
       → Moiré 검출 (FFT)
       → rPPG 검출 (POS + CHROM)
       → Spoof 판정 (multi-ROI, SNR, background)
       → Face Recognition → 환영/거부
```

## 설치
```powershell
pip install -r requirements.txt
```

첫 실행 시 MediaPipe Face Landmarker 모델(`face_landmarker.task`, ~3MB)이 `models/` 폴더로 자동 다운로드됩니다.

> **Windows 설치 팁**: `dlib`은 소스 빌드 시 CMake/VC++가 필요하고 한국어 환경에서 cp949 인코딩 에러가 잦아서, requirements에 `dlib-bin` (prebuilt wheel)을 명시했습니다. 그래도 막히면:
> ```powershell
> pip install dlib-bin face-recognition-models
> pip install --no-deps face_recognition
> ```
> MediaPipe는 0.10.15 이상이 필요합니다 (Tasks API 사용).

## 사용
```powershell
# GUI 데모 (웹캠 실시간)
python gui_app.py

# CLI 데모 (빠른 테스트)
python demo_cli.py

# 평가 (데이터 일괄 처리)
python evaluate.py --data data/
```

## 디렉터리 구조
```
liveness/
├── face_detector.py        MediaPipe ROI 추출
├── pre_screen.py           사전 거부 필터
├── rppg/                   rPPG 신호 추출 + spoof 판정
├── moire/                  Moiré 패턴 검출
├── face_auth/              얼굴 등록/인식
├── liveness_pipeline.py    liveness 통합
├── auth_pipeline.py        liveness + face auth 통합
├── gui_app.py              tkinter GUI
├── demo_cli.py             CLI 데모
├── evaluate.py             평가 스크립트
└── data/                   촬영한 영상 (live/photo/screen) + 등록 사진
```

## 참고문헌
- POS: Wang et al., "Algorithmic Principles of Remote PPG" (TBME 2017)
- CHROM: de Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG" (TBME 2013)
- Moiré: Patel, Han, Jain, "Use of Moiré Patterns to Detect Replay Video Attacks" (ICB 2015)

## 웹 제어판 (권장 UI)

느리고 투박한 tkinter GUI 대신, 현대적인 HTML 제어판을 제공합니다. CV/ML 분석(rPPG·Moiré·얼굴인식)은 전부 Python 백엔드에서 수행하고, 브라우저는 카메라 입력과 화면 표시만 담당합니다.

### 실행
```powershell
python run_web.py
```
그 후 브라우저에서 **http://127.0.0.1:5000** 을 엽니다. 카메라는 브라우저의 `getUserMedia` 로 동작하므로, 권한 요청이 뜨면 허용해 주세요. 캡처한 프레임만 Flask 서버로 전송되어 분석됩니다.

### 패널 설명
- **상태**: 파이프라인 로드 여부, Moiré 모델 학습 여부, 등록자 수를 표시합니다.
- **얼굴 등록**: 이름을 입력하고 현재 화면을 캡처해 등록합니다.
- **등록자 목록**: 등록된 사용자 확인 및 삭제.
- **인증**: 3초 준비 후 약 10초 촬영 → 라이브니스 + 얼굴인식. 결과(환영/스푸핑 차단/미등록/얼굴 없음)와 한국어 사유, 상세 수치를 보여 줍니다. *발표 모드*를 켜면 rPPG는 경고만 하고 차단하지 않습니다.
- **얼굴 감지 테스트**: 등록 없이 라이브니스만 검사합니다.
- **데이터 수집 (Moiré 학습용)**: 라벨(실제 얼굴=live / 사진=photo / 화면영상=screen)을 고르고 약 5초 녹화해 `data/<라벨>/` 에 저장합니다.
- **Moiré 모델 학습**: 수집한 데이터로 LBP+SVM 모델을 학습합니다.

### Moiré 학습 워크플로
1. **데이터 수집** — 실제 얼굴(live), 사진(photo), 화면 재생(screen) 샘플을 각 클래스별로 수집합니다. (권장: 클래스당 5개 이상)
2. **학습** — *Moiré 모델 학습*의 *학습 시작* 버튼을 누르면 `models/moire_lbp_svm.pkl` 이 생성됩니다.
3. **자동 적용** — 이후 인증/라이브니스 검사 시 학습된 Moiré 모델이 자동으로 함께 사용됩니다.
