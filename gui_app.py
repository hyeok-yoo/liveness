"""Face Liveness + Auth Demo — tkinter GUI."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, simpledialog

import cv2
from PIL import Image, ImageTk

from auth_pipeline import AuthPipeline, AuthResult
from face_auth import enroll_face_from_array, list_enrolled

CAPTURE_SECONDS = 10
PREP_SECONDS = 3  # 자세 잡을 준비 시간 (캡처 시작 전)
PREVIEW_INTERVAL_MS = 33  # ~30 fps


class LivenessApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Liveness + Auth Demo")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.current_frame_bgr = None
        self.capture_buffer: list | None = None
        self.is_busy = False
        self._tk_image = None  # prevent GC
        self._countdown_remaining = 0
        self.demo_mode_var = tk.BooleanVar(value=False)  # 발표용 soft 모드 토글

        self._build_ui()
        self._set_status("초기화 중...", "gray")

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self._set_status("카메라를 찾을 수 없습니다. 웹캠 연결 확인하세요.", "red")

        # AuthPipeline 초기화 — 무거운 작업이므로 백그라운드에서
        threading.Thread(target=self._init_pipeline, daemon=True).start()

        self._schedule_preview()

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # 카메라 프리뷰
        self.preview_label = tk.Label(
            self.root, width=640, height=480, bg="black"
        )
        self.preview_label.pack()

        # 상태 표시줄
        self.status_var = tk.StringVar(value="대기 중")
        status_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        status_frame.pack(fill=tk.X)
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
            padx=8,
            pady=4,
            font=("맑은 고딕", 12),
        )
        self.status_label.pack(fill=tk.X)

        # 버튼 영역
        btn_frame = tk.Frame(self.root, pady=6)
        btn_frame.pack()

        self.btn_enroll = tk.Button(
            btn_frame, text="등록", width=10, command=self.on_enroll
        )
        self.btn_enroll.pack(side=tk.LEFT, padx=6)

        self.btn_auth = tk.Button(
            btn_frame, text="인증", width=10, command=self.on_authenticate
        )
        self.btn_auth.pack(side=tk.LEFT, padx=6)

        self.btn_list = tk.Button(
            btn_frame, text="등록자 보기", width=12, command=self.on_show_enrolled
        )
        self.btn_list.pack(side=tk.LEFT, padx=6)

        # 발표용 soft mode 체크박스
        self.chk_demo = tk.Checkbutton(
            btn_frame,
            text="발표 모드 (rPPG 경고만)",
            variable=self.demo_mode_var,
            command=self._on_demo_toggle,
        )
        self.chk_demo.pack(side=tk.LEFT, padx=12)

    def _on_demo_toggle(self) -> None:
        """발표 모드 토글 시 파이프라인 재초기화."""
        if hasattr(self, "auth"):
            try:
                self.auth.close()
            except Exception:  # noqa: BLE001
                pass
            del self.auth
        self._set_status("모드 변경 중...", "gray")
        threading.Thread(target=self._init_pipeline, daemon=True).start()

    # ------------------------------------------------------------------
    # 파이프라인 초기화 (백그라운드)
    # ------------------------------------------------------------------

    def _init_pipeline(self) -> None:
        try:
            from liveness_pipeline import LivenessPipeline
            from face_auth import FaceRecognizer
            # 발표용 demo 모드 토글에 따라 rppg_decisive 설정
            decisive = not self.demo_mode_var.get()
            liveness = LivenessPipeline(rppg_decisive=decisive)
            self.auth = AuthPipeline(liveness=liveness, recognizer=FaceRecognizer())
            self.root.after(0, self._set_status, "대기 중", "black")
        except Exception as exc:  # noqa: BLE001
            self.root.after(
                0, self._set_status, f"파이프라인 초기화 실패: {exc}", "red"
            )

    # ------------------------------------------------------------------
    # 프리뷰 루프
    # ------------------------------------------------------------------

    def _schedule_preview(self) -> None:
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame_bgr = frame
                    if self.capture_buffer is not None:
                        self.capture_buffer.append(frame.copy())
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    self._tk_image = ImageTk.PhotoImage(img)
                    self.preview_label.configure(image=self._tk_image)
        except Exception as exc:  # noqa: BLE001
            print(f"[preview] {exc}")

        self.root.after(PREVIEW_INTERVAL_MS, self._schedule_preview)

    # ------------------------------------------------------------------
    # [인증] 버튼
    # ------------------------------------------------------------------

    def on_authenticate(self) -> None:
        if self.is_busy:
            return
        if not hasattr(self, "auth"):
            self._set_status("아직 초기화 중입니다. 잠시 후 다시 시도하세요.", "gray")
            return
        self.is_busy = True
        self._set_buttons_state(tk.DISABLED)

        # 준비 카운트다운: 사용자가 정면 응시하도록 자세 잡을 시간
        self._prep_remaining = PREP_SECONDS
        self._tick_prep()

    def _tick_prep(self) -> None:
        if self._prep_remaining > 0:
            self._set_status(
                f"준비 중... {self._prep_remaining}초 후 캡처 시작 — 카메라 정면 응시!",
                "orange",
            )
            self._prep_remaining -= 1
            self.root.after(1000, self._tick_prep)
        else:
            # 캡처 시작
            self.capture_buffer = []
            self._countdown_remaining = CAPTURE_SECONDS
            self._tick_countdown()

    def _tick_countdown(self) -> None:
        if self._countdown_remaining > 0:
            self._set_status(
                f"촬영 중... 남은 시간: {self._countdown_remaining}초", "blue"
            )
            self._countdown_remaining -= 1
            self.root.after(1000, self._tick_countdown)
        else:
            # 캡처 완료 → 백그라운드 분석
            frames = self.capture_buffer
            self.capture_buffer = None
            self._set_status("분석 중...", "blue")
            threading.Thread(
                target=self._run_auth, args=(frames,), daemon=True
            ).start()

    def _run_auth(self, frames: list) -> None:
        try:
            result: AuthResult = self.auth.authenticate_frames(frames)
            self.root.after(0, self._show_auth_result, result)
        except Exception as exc:  # noqa: BLE001
            self.root.after(
                0, self._set_status, f"인증 오류: {exc}", "red"
            )
            self.root.after(0, self._finish_busy)

    def _show_auth_result(self, result: AuthResult) -> None:
        stage_color = {
            "match": "green",
            "spoof_blocked": "red",
            "no_match": "orange",
            "no_face": "gray",
        }
        color = stage_color.get(result.stage, "black")
        self._set_status(result.message, color)
        self._finish_busy()

    # ------------------------------------------------------------------
    # [등록] 버튼
    # ------------------------------------------------------------------

    def on_enroll(self) -> None:
        if self.is_busy:
            return
        name = simpledialog.askstring("얼굴 등록", "이름을 입력하세요:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()

        frame = self.current_frame_bgr
        if frame is None:
            self._set_status("카메라 프레임을 가져올 수 없습니다.", "red")
            return

        self.is_busy = True
        self._set_buttons_state(tk.DISABLED)
        self._set_status(f"{name} 등록 중...", "blue")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        threading.Thread(
            target=self._run_enroll, args=(name, rgb), daemon=True
        ).start()

    def _run_enroll(self, name: str, rgb_frame) -> None:
        try:
            enroll_face_from_array(name, rgb_frame)
            self.root.after(
                0, self._set_status, f"{name} 등록 완료", "green"
            )
        except ValueError as exc:
            self.root.after(
                0, self._set_status, f"등록 실패: {exc}", "red"
            )
        except Exception as exc:  # noqa: BLE001
            self.root.after(
                0, self._set_status, f"등록 오류: {exc}", "red"
            )
        finally:
            self.root.after(0, self._finish_busy)

    # ------------------------------------------------------------------
    # [등록자 보기] 버튼
    # ------------------------------------------------------------------

    def on_show_enrolled(self) -> None:
        enrolled = list_enrolled()
        if enrolled:
            msg = "\n".join(enrolled)
        else:
            msg = "등록된 사용자 없음"
        messagebox.showinfo("등록자 목록", msg, parent=self.root)

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    def _set_status(self, text: str, color: str = "black") -> None:
        self.status_var.set(f"Status: {text}")
        self.status_label.configure(fg=color)

    def _set_buttons_state(self, state: str) -> None:
        for btn in (self.btn_enroll, self.btn_auth, self.btn_list):
            btn.configure(state=state)

    def _finish_busy(self) -> None:
        self.is_busy = False
        self._set_buttons_state(tk.NORMAL)

    def _on_close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "auth"):
            try:
                self.auth.close()
            except Exception:  # noqa: BLE001
                pass
        self.root.destroy()


# ------------------------------------------------------------------
# 진입점
# ------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    LivenessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
