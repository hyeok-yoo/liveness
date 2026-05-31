"""웹 제어판 패키지.

무거운 CV/ML 파이프라인(mediapipe / face_recognition 등)은 이 패키지를
import 하는 시점에 불러오지 않는다. 모든 무거운 import 는 라우트 핸들러나
지연 로더 내부에서 수행한다.
"""
