"use strict";

// ---------------------------------------------------------------------------
// DOM 참조
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const preview = $("preview");
const overlay = $("overlay");
const overlayText = $("overlay-text");
const statusChip = $("status-chip");
const camError = $("cam-error");
const captureCanvas = $("capture-canvas");
const ctx = captureCanvas.getContext("2d");

let cameraReady = false;
let busy = false; // 전역 캡처 진행 중 플래그

// 캡처에 영향을 주는 버튼들 (촬영 중 비활성화)
const captureButtons = ["btn-auth", "btn-liveness", "btn-collect", "btn-enroll", "btn-train"];

// ---------------------------------------------------------------------------
// 카메라 초기화
// ---------------------------------------------------------------------------
async function initCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCamError("이 브라우저는 카메라 접근(getUserMedia)을 지원하지 않습니다.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    preview.srcObject = stream;
    cameraReady = true;
    setChip("카메라 작동 중", "live");
    camError.classList.add("hidden");
  } catch (err) {
    cameraReady = false;
    let msg = "카메라에 접근할 수 없습니다.";
    if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
      msg = "카메라 권한이 거부되었습니다. 브라우저 설정에서 권한을 허용해 주세요.";
    } else if (err && err.name === "NotFoundError") {
      msg = "사용 가능한 카메라를 찾을 수 없습니다.";
    }
    showCamError(msg);
    setChip("카메라 사용 불가", "");
  }
}

function showCamError(msg) {
  camError.textContent = msg;
  camError.classList.remove("hidden");
}

function setChip(text, cls) {
  statusChip.textContent = text;
  statusChip.className = "status-chip" + (cls ? " " + cls : "");
}

// ---------------------------------------------------------------------------
// 프레임 캡처 헬퍼
// ---------------------------------------------------------------------------
function grabFrameDataUrl() {
  // 미리보기는 거울처럼 좌우 반전돼 보이지만, 분석용 캔버스는 원본 방향으로 그린다.
  ctx.drawImage(preview, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", 0.7);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * 준비 카운트다운 후 일정 시간 동안 목표 fps 로 JPEG dataURL 프레임을 모은다.
 * 반환: { frames: string[], fps: number(실측) }
 */
async function captureFrames({ prepSeconds = 3, captureSeconds = 10, targetFps = 15, onTick }) {
  if (!cameraReady) {
    throw new Error("카메라가 준비되지 않았습니다.");
  }
  // 준비 카운트다운
  overlay.classList.remove("hidden");
  for (let s = prepSeconds; s > 0; s--) {
    overlayText.textContent = String(s);
    await sleep(1000);
  }

  const frames = [];
  const interval = 1000 / targetFps;
  const start = performance.now();
  const endAt = start + captureSeconds * 1000;
  let nextAt = start;

  while (performance.now() < endAt) {
    const now = performance.now();
    if (now >= nextAt) {
      frames.push(grabFrameDataUrl());
      nextAt += interval;
      const remain = Math.max(0, Math.ceil((endAt - now) / 1000));
      overlayText.textContent = "촬영 중…\n남은 " + remain + "초";
      if (onTick) onTick(remain);
    }
    await sleep(5);
  }

  const elapsed = (performance.now() - start) / 1000;
  const fps = elapsed > 0 ? frames.length / elapsed : targetFps;
  overlay.classList.add("hidden");
  return { frames, fps };
}

// ---------------------------------------------------------------------------
// UI 유틸
// ---------------------------------------------------------------------------
function setBusy(on, chipText) {
  busy = on;
  captureButtons.forEach((id) => {
    const el = $(id);
    if (el) el.disabled = on;
  });
  if (on) setChip(chipText || "처리 중…", "busy");
  else setChip(cameraReady ? "카메라 작동 중" : "카메라 사용 불가", cameraReady ? "live" : "");
}

function spinnerBtn(btn, label) {
  btn.dataset.orig = btn.innerHTML;
  btn.innerHTML = '<span class="spinner"></span>' + (label || "");
}
function restoreBtn(btn) {
  if (btn.dataset.orig !== undefined) btn.innerHTML = btn.dataset.orig;
}

function setMsg(el, text, cls) {
  el.textContent = text || "";
  el.className = "inline-msg" + (cls ? " " + cls : "");
}

let toastTimer = null;
function toast(text, cls) {
  const t = $("toast");
  t.textContent = text;
  t.className = "toast" + (cls ? " " + cls : "");
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 3200);
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v === "number") return v.toFixed(digits);
  return String(v);
}

// ---------------------------------------------------------------------------
// 상태 / 등록자
// ---------------------------------------------------------------------------
async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const s = await res.json();
    setBadge($("st-pipeline"), s.pipeline_loaded, s.pipeline_loaded ? "예" : "미로드", true);
    setBadge($("st-moire"), s.moire_model_trained, s.moire_model_trained ? "예" : "아니오");
    const n = (s.enrolled || []).length;
    $("st-enrolled").textContent = n + "명";
    $("st-enrolled").className = "badge badge-num";
    const c = s.sample_counts || {};
    $("cnt-live").textContent = c.live ?? 0;
    $("cnt-photo").textContent = c.photo ?? 0;
    $("cnt-screen").textContent = c.screen ?? 0;
    renderEnrolled(s.enrolled || []);
  } catch (e) {
    /* 상태 조회 실패는 조용히 무시 */
  }
}

function setBadge(el, good, text, neutralOk) {
  el.textContent = text;
  if (good) el.className = "badge badge-ok";
  else el.className = "badge " + (neutralOk ? "badge-muted" : "badge-no");
}

function renderEnrolled(names) {
  const list = $("enrolled-list");
  list.innerHTML = "";
  if (!names.length) {
    list.innerHTML = '<li class="empty">등록된 사용자가 없습니다.</li>';
    return;
  }
  names.forEach((name) => {
    const li = document.createElement("li");
    const span = document.createElement("span");
    span.textContent = name;
    const btn = document.createElement("button");
    btn.className = "btn btn-danger";
    btn.type = "button";
    btn.textContent = "삭제";
    btn.addEventListener("click", () => removeEnrolled(name, btn));
    li.appendChild(span);
    li.appendChild(btn);
    list.appendChild(li);
  });
}

async function removeEnrolled(name, btn) {
  btn.disabled = true;
  try {
    const r = await postJSON("/api/remove_enrolled", { name });
    if (r.ok) {
      toast("'" + name + "' 삭제됨", "ok");
      refreshStatus();
    } else {
      toast("삭제 실패", "err");
      btn.disabled = false;
    }
  } catch (e) {
    toast("삭제 중 오류", "err");
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 얼굴 등록
// ---------------------------------------------------------------------------
async function doEnroll() {
  const btn = $("btn-enroll");
  const name = $("enroll-name").value.trim();
  const msg = $("enroll-msg");
  if (!name) {
    setMsg(msg, "이름을 입력해 주세요.", "err");
    return;
  }
  if (!cameraReady) {
    setMsg(msg, "카메라가 준비되지 않았습니다.", "err");
    return;
  }
  const image = grabFrameDataUrl();
  spinnerBtn(btn, "등록 중");
  btn.disabled = true;
  setMsg(msg, "현재 화면으로 등록 중…", "info");
  try {
    const r = await postJSON("/api/enroll", { name, image });
    setMsg(msg, r.message, r.ok ? "ok" : "err");
    if (r.ok) {
      $("enroll-name").value = "";
      refreshStatus();
    }
  } catch (e) {
    setMsg(msg, "등록 중 오류가 발생했습니다.", "err");
  } finally {
    restoreBtn(btn);
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 인증
// ---------------------------------------------------------------------------
const STAGE_TITLE = {
  match: (name) => "환영합니다, " + (name || "사용자") + "님",
  spoof_blocked: () => "스푸핑 차단됨",
  no_match: () => "등록되지 않은 사용자",
  no_face: () => "얼굴을 찾지 못했습니다",
};

function showBanner(stage, titleText, reasons, details) {
  const banner = $("result-banner");
  banner.className = "result-banner " + (stage || "neutral");
  $("result-title").textContent = titleText;
  const ul = $("result-reasons");
  ul.innerHTML = "";
  (reasons || []).forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    ul.appendChild(li);
  });
  renderDetails(details);
  banner.classList.remove("hidden");
  banner.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderDetails(d) {
  const body = $("result-details-body");
  const det = $("result-details");
  if (!d || Object.keys(d).length === 0) {
    det.classList.add("hidden");
    return;
  }
  det.classList.remove("hidden");
  const rows = [
    ["라이브니스 판정", d.is_live === undefined ? "—" : d.is_live ? "실제" : "스푸핑"],
    ["검출 프레임", (d.frames_with_face ?? "—") + " / " + (d.frames_total ?? "—")],
    ["심박수 (BPM)", fmt(d.fused_bpm, 1)],
    ["얼굴 SNR (dB)", fmt(d.fused_snr_db, 2)],
    ["배경 SNR (dB)", fmt(d.background_snr_db, 2)],
    ["Moiré 점수", fmt(d.moire_score, 4)],
    ["Moiré 임계값", fmt(d.moire_threshold, 4)],
  ];
  body.innerHTML = "";
  rows.forEach(([k, v]) => {
    const kd = document.createElement("div");
    kd.className = "k";
    kd.textContent = k;
    const vd = document.createElement("div");
    vd.className = "v";
    vd.textContent = v;
    body.appendChild(kd);
    body.appendChild(vd);
  });
}

async function doAuthenticate() {
  if (busy) return;
  const btn = $("btn-auth");
  const demo = $("demo-mode").checked;
  spinnerBtn(btn, "촬영 중");
  setBusy(true, "준비 중…");
  $("result-banner").classList.add("hidden");
  try {
    const { frames, fps } = await captureFrames({
      prepSeconds: 0,
      captureSeconds: 6,
      targetFps: 15,
    });
    setBusy(true, "분석 중…");
    btn.innerHTML = '<span class="spinner"></span>분석 중';
    const r = await postJSON("/api/authenticate", { frames, fps, demo_mode: demo });
    const titleFn = STAGE_TITLE[r.stage] || (() => r.message || "결과");
    showBanner(r.stage, titleFn(r.name), r.korean_reasons, r.details);
  } catch (e) {
    showBanner("neutral", "오류: " + (e.message || e), [], null);
  } finally {
    restoreBtn(btn);
    setBusy(false);
    refreshStatus();
  }
}

// ---------------------------------------------------------------------------
// 얼굴 감지 테스트 (liveness-only)
// ---------------------------------------------------------------------------
async function doLiveness() {
  if (busy) return;
  const btn = $("btn-liveness");
  const demo = $("demo-mode").checked;
  spinnerBtn(btn, "촬영 중");
  setBusy(true, "준비 중…");
  $("result-banner").classList.add("hidden");
  try {
    const { frames, fps } = await captureFrames({
      prepSeconds: 0,
      captureSeconds: 6,
      targetFps: 15,
    });
    setBusy(true, "분석 중…");
    const r = await postJSON("/api/liveness", { frames, fps, demo_mode: demo });
    const stage = r.is_live ? "match" : "spoof_blocked";
    const title = r.is_live ? "실제 얼굴로 판정됨" : "스푸핑으로 판정됨";
    showBanner(stage, title, r.korean_reasons, r.details);
  } catch (e) {
    showBanner("neutral", "오류: " + (e.message || e), [], null);
  } finally {
    restoreBtn(btn);
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// 데이터 수집
// ---------------------------------------------------------------------------
async function doCollect() {
  if (busy) return;
  const btn = $("btn-collect");
  const label = $("collect-label").value;
  const msg = $("collect-msg");
  spinnerBtn(btn, "녹화 중");
  setBusy(true, "녹화 준비 중…");
  setMsg(msg, "", "");
  try {
    const { frames, fps } = await captureFrames({
      prepSeconds: 3,
      captureSeconds: 5,
      targetFps: 15,
    });
    setBusy(true, "저장 중…");
    const r = await postJSON("/api/collect_sample", { label, frames, fps });
    if (r.ok) {
      setMsg(msg, r.n_frames + "프레임 저장됨: " + r.path, "ok");
      toast("샘플 저장 완료", "ok");
      if (r.sample_counts) {
        $("cnt-live").textContent = r.sample_counts.live;
        $("cnt-photo").textContent = r.sample_counts.photo;
        $("cnt-screen").textContent = r.sample_counts.screen;
      }
    } else {
      setMsg(msg, r.message || "저장 실패", "err");
    }
  } catch (e) {
    setMsg(msg, "녹화 중 오류: " + (e.message || e), "err");
  } finally {
    restoreBtn(btn);
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Moiré 학습
// ---------------------------------------------------------------------------
async function doTrain() {
  const btn = $("btn-train");
  const msg = $("train-msg");
  spinnerBtn(btn, "학습 중");
  btn.disabled = true;
  setMsg(msg, "모델 학습 중… (영상 수에 따라 시간이 걸릴 수 있습니다)", "info");
  try {
    const r = await postJSON("/api/train_moire", {});
    if (r.ok) {
      const acc = r.train_accuracy == null ? "—" : (r.train_accuracy * 100).toFixed(1) + "%";
      setMsg(
        msg,
        "학습 완료 — live " + r.n_live + " · spoof " + r.n_spoof + " · 학습 정확도 " + acc + ". " + (r.message || ""),
        "ok"
      );
      toast("Moiré 모델 학습 완료", "ok");
      refreshStatus();
    } else {
      setMsg(msg, r.message || "학습 실패", "err");
    }
  } catch (e) {
    setMsg(msg, "학습 중 오류가 발생했습니다.", "err");
  } finally {
    restoreBtn(btn);
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// 테마 토글
// ---------------------------------------------------------------------------
function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved === "dark") document.documentElement.setAttribute("data-theme", "dark");
  updateThemeIcon();
  $("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    if (cur === "dark") {
      document.documentElement.removeAttribute("data-theme");
      localStorage.setItem("theme", "light");
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
      localStorage.setItem("theme", "dark");
    }
    updateThemeIcon();
  });
}
function updateThemeIcon() {
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  $("theme-toggle").textContent = dark ? "☀️" : "🌙";
}

// ---------------------------------------------------------------------------
// 부팅
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initCamera();
  refreshStatus();
  $("btn-enroll").addEventListener("click", doEnroll);
  $("btn-auth").addEventListener("click", doAuthenticate);
  $("btn-liveness").addEventListener("click", doLiveness);
  $("btn-collect").addEventListener("click", doCollect);
  $("btn-train").addEventListener("click", doTrain);
  $("enroll-name").addEventListener("keydown", (e) => {
    if (e.key === "Enter") doEnroll();
  });
});
