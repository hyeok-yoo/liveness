"""웹 제어판 실행 진입점.

사용법:
    python run_web.py
그 후 브라우저에서 http://127.0.0.1:5000 을 연다.
"""
from webapp.app import app, main


if __name__ == "__main__":
    main()
