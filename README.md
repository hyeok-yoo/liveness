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
