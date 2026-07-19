const appConfig = window.APP_CONFIG || {};
const TEST_SKIP_ENABLED = Boolean(appConfig.enableTestSkip);
const LIGHT_SETTLE_MS = Number(appConfig.lightSettleMs || 180);
// Flash pacing floor: ≥250ms per phase keeps the strobe at ≤2 flashes/s —
// safely under the WCAG 2.3.1 photosensitivity limit of 3 flashes/s — and
// gives camera auto-exposure time to react to each change.
const FLASH_PHASE_MS = Number(appConfig.flashPhaseMs || 250);
const FLASH_FRAME_WAIT_MAX_MS = Number(appConfig.flashFrameWaitMaxMs || 400);
// Audio: mic level preflight runs before the phrase is disclosed.
const MIC_PREFLIGHT_MS = Number(appConfig.micPreflightMs || 2000);
const MIC_PREFLIGHT_TRIES = 3;
const MIC_MIN_RMS = Number(appConfig.micMinRms || 0.01);
const LIGHT_SAMPLE_COUNT = Number(appConfig.lightSampleCount || 4);
const LIGHT_SAMPLE_INTERVAL_MS = Number(appConfig.lightSampleIntervalMs || 70);
// Generous ceiling: the first rPPG call may wait out the ~1min model warmup
// on the server. Without a timeout a dropped tunnel hangs the flow forever.
const REQUEST_TIMEOUT_MS = Number(appConfig.requestTimeoutMs || 120000);

const FLOW = [
  {
    id: "camera",
    title: "Проверка камеры",
    value: "камера",
    hint: "Разрешите доступ к камере и совместите лицо с овалом.",
    action: "Включить камеру",
  },
  {
    id: "active_light",
    title: "Проверка активным светом",
    value: "цветные вспышки",
    hint: "Держите лицо внутри овала, пока экран мигает цветными вспышками.",
    action: "Запустить вспышки",
  },
  {
    id: "gesture",
    title: "Проверка жестом",
    value: "жест",
    hint: "Выполните показанный жест, удерживая лицо внутри овала.",
    action: "Подтвердить жест",
  },
  {
    id: "rppg",
    title: "Проверка пульса",
    value: "пульс",
    hint: "Не двигайтесь, удерживая лицо внутри овала, для короткого замера rPPG.",
    action: "Замерить пульс",
  },
  {
    id: "audio",
    title: "Проверка аудио-фразой",
    value: "фраза",
    hint: "Произнесите сгенерированную фразу. Транскрипт распознаётся на сервере.",
    action: "Записать аудио",
  },
  {
    id: "score",
    title: "Оценка риска",
    value: "оценка",
    hint: "Отправить собранные данные в ML-сервис и колбэк risk-api.",
    action: "Отправить на оценку",
  },
];

const state = {
  session: null,
  stream: null,
  pulseTimer: null,
  faceTimer: null,
  pulseSamples: [],
  suppressPulseCollection: false,
  facePresent: null,
  faceConfidence: null,
  gestureAttempt: null,
  serviceEvidence: {},
  stepIndex: 0,
  scored: false,
  skipped: new Set(),
  stepStatus: {},
  expectedLuma: [],
  observedLuma: [],
  pulse: null,
  audio: null,
};

const el = {
  apiUrl: document.querySelector("#apiUrl"),
  uid: document.querySelector("#uid"),
  checkId: document.querySelector("#checkId"),
  status: document.querySelector("#status"),
  progressDots: document.querySelector("#progressDots"),
  primaryAction: document.querySelector("#primaryAction"),
  skipStep: document.querySelector("#skipStep"),
  resetFlow: document.querySelector("#resetFlow"),
  stage: document.querySelector("#challengeStage"),
  stageOverlay: document.querySelector(".stage-overlay"),
  flashFullscreen: document.querySelector("#flashFullscreen"),
  camera: document.querySelector("#camera"),
  faceGuide: document.querySelector("#faceGuide"),
  faceOval: document.querySelector("#faceOval"),
  guideHint: document.querySelector("#guideHint"),
  currentStep: document.querySelector("#currentStep"),
  stageValue: document.querySelector("#stageValue"),
  stepHint: document.querySelector("#stepHint"),
  decision: document.querySelector("#decision"),
  riskLine: document.querySelector("#riskLine"),
  checksBreakdown: document.querySelector("#checksBreakdown"),
  scoreJson: document.querySelector("#scoreJson"),
};

el.apiUrl.value = appConfig.mlApiUrl || "http://localhost:8100";
el.uid.value = `user-${Math.random().toString(16).slice(2, 8)}`;
el.checkId.value = `check-${Date.now()}`;
el.skipStep.style.display = TEST_SKIP_ENABLED ? "block" : "none";

function setStatus(value) {
  el.status.textContent = value;
}

// The server returns English status/decision tokens (passed/failed/allow/…);
// map them to Russian for display while keeping the raw token for CSS classes
// and flow logic.
const STATUS_RU = {
  passed: "пройдено",
  failed: "не пройдено",
  unknown: "неизвестно",
  skipped: "пропущено",
  pending: "ожидание",
  allow: "разрешено",
  review: "проверка",
  deny: "отказано",
};

function statusRu(value) {
  return STATUS_RU[value] || value;
}

// Check identifiers from the server, mapped to Russian labels for the score
// breakdown; unknown names fall through unchanged.
const CHECK_NAME_RU = {
  active_light: "Активный свет",
  gesture: "Жест",
  audio: "Аудио",
  rppg: "Пульс",
  classifier: "Видео-классификатор",
};

function checkNameRu(value) {
  return CHECK_NAME_RU[value] || value;
}

// Diagnostic "reason" sentences come from the server in English. They render in
// the log panel and as tooltips, so map the known ones to Russian; dynamic or
// unknown messages (e.g. "… inference failed: RuntimeError") fall through.
const REASON_RU = {
  "frame classifier cannot pass without a detected face": "классификатор кадров не может пройти без обнаруженного лица",
  "frame classifier evidence is missing": "данные классификатора кадров отсутствуют",
  "AI restoration/upscaling detected on the input — rejected": "обнаружено ИИ-восстановление/апскейл входных данных — отклонено",
  "low-detail input — fake verdict withheld (signal unreliable)": "мало деталей во входных данных — вердикт о подделке отложен (сигнал ненадёжен)",
  "deepfake classifier probability evaluated": "оценена вероятность дипфейка классификатором",
  "active light evidence is missing": "данные активного света отсутствуют",
  "active light cannot pass without a detected face": "активный свет не может пройти без обнаруженного лица",
  "face flashing frame-pair verifier evaluated": "проверены пары кадров вспышек света",
  "not enough active light samples": "недостаточно образцов активного света",
  "active light correlation is undefined": "корреляция активного света не определена",
  "face luminance response compared with screen challenge": "яркость лица сопоставлена с вспышками экрана",
  "rPPG evidence is missing": "данные rPPG отсутствуют",
  "rPPG cannot pass without a detected face ROI": "rPPG не может пройти без обнаруженной области лица",
  "rPPG raw samples are required": "требуются исходные образцы rPPG",
  "rPPG sample window is too short": "окно образцов rPPG слишком короткое",
  "rPPG signal quality is unavailable": "качество сигнала rPPG недоступно",
  "physiological pulse signal evaluated with sliding-window stability": "физиологический сигнал пульса оценён со стабильностью скользящего окна",
  "rPPG signal quality is too low": "качество сигнала rPPG слишком низкое",
  "rPPG model heart-rate and signal-quality output evaluated": "оценены пульс и качество сигнала моделью rPPG",
  "gesture evidence is missing": "данные жеста отсутствуют",
  "gesture requires a real detector, manual confirmation is not accepted": "жест требует реального детектора, ручное подтверждение не принимается",
  "gesture cannot pass without a detected face/body target": "жест не может пройти без обнаруженной цели лица/тела",
  "gesture challenge response evaluated": "оценён ответ на проверку жестом",
  "audio evidence is missing": "данные аудио отсутствуют",
  "audio anti-spoof model result is required": "требуется результат анти-спуфинг модели аудио",
  "audio phrase transcript is unavailable": "транскрипт аудио-фразы недоступен",
  "audio challenge and synthetic speech signals evaluated": "оценены аудио-проверка и признаки синтетической речи",
  "check skipped in test mode": "проверка пропущена в тестовом режиме",
  "video classifier model is not configured": "модель видео-классификатора не настроена",
  "audio anti-spoof model is not configured": "анти-спуфинг модель аудио не настроена",
};

function reasonRu(value) {
  return value == null ? value : REASON_RU[value] || value;
}

// A score factor is "<check name>: <reason>" (or "…: insufficient evidence").
// Localize the check name and the reason halves independently.
function factorRu(value) {
  if (typeof value !== "string") return value;
  const separator = value.indexOf(": ");
  if (separator === -1) return value;
  const name = value.slice(0, separator);
  const detail = value.slice(separator + 2);
  const detailRu = detail === "insufficient evidence" ? "недостаточно данных" : reasonRu(detail);
  return `${checkNameRu(name)}: ${detailRu}`;
}

function api(path) {
  return `${el.apiUrl.value.replace(/\/$/, "")}${path}`;
}

function currentFlowStep() {
  return FLOW[state.stepIndex];
}

function getStep(type) {
  return state.session.challenge.steps.find((step) => step.type === type);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchChecked(path, options) {
  let response;
  try {
    response = await fetch(api(path), { ...options, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
  } catch (error) {
    if (error?.name === "TimeoutError" || error?.name === "AbortError") {
      logLine(`${path}: нет ответа за ${Math.round(REQUEST_TIMEOUT_MS / 1000)}с — сервер или туннель недоступен`);
      throw new Error(`Нет ответа от ${path} за ${Math.round(REQUEST_TIMEOUT_MS / 1000)}с`);
    }
    logLine(`${path}: сетевая ошибка — ${error?.message || error}`);
    throw error;
  }
  if (!response.ok) {
    const detail = await response.text();
    logLine(`${path}: HTTP ${response.status}`);
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

async function requestJson(path, options = {}) {
  return fetchChecked(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
}

async function requestForm(path, form) {
  return fetchChecked(path, { method: "POST", body: form });
}

// The single CTA under the stage: red "start" before a session (and for a
// rerun after scoring — the server session is consumed, a new one is needed),
// dark per-step action while a flow is running.
function setPrimaryAction(label, { start = false } = {}) {
  el.primaryAction.textContent = label;
  el.primaryAction.classList.toggle("btn-primary", start);
  el.primaryAction.classList.toggle("btn-dark", !start);
}

function renderProgress() {
  const dots = el.progressDots.children;
  for (let index = 0; index < dots.length; index += 1) {
    const done = state.session && (index < state.stepIndex || (state.scored && index === state.stepIndex));
    dots[index].className = done ? "done" : state.session && index === state.stepIndex ? "active" : "";
  }
}

// Re-trigger the overlay's slide-in only when the visible step actually
// changes, so mid-step re-renders don't flicker.
let lastStepKey = null;

function animateStepSwap() {
  const key = state.session ? `step-${state.stepIndex}` : "idle";
  if (key === lastStepKey) return;
  lastStepKey = key;
  el.stageOverlay.classList.remove("swap");
  void el.stageOverlay.offsetWidth;
  el.stageOverlay.classList.add("swap");
}

function renderStep() {
  const step = currentFlowStep();
  renderProgress();
  animateStepSwap();
  if (!state.session) {
    el.primaryAction.disabled = false;
    setPrimaryAction("Начать проверку", { start: true });
    el.skipStep.disabled = true;
    el.currentStep.textContent = "Проверка не запущена";
    el.stageValue.textContent = "Готовы?";
    el.stepHint.textContent = "Нажмите «Начать проверку» — понадобится доступ к камере и микрофону.";
    return;
  }

  el.primaryAction.disabled = false;
  el.skipStep.disabled = !TEST_SKIP_ENABLED || step.id === "score";
  el.currentStep.textContent = `Шаг ${state.stepIndex + 1}/${FLOW.length} — ${step.title}`;
  el.stageValue.textContent = displayValue(step);
  el.stepHint.textContent = displayHint(step);
  if (step.id === "score" && state.scored) {
    setPrimaryAction("Пройти ещё раз", { start: true });
  } else {
    setPrimaryAction(step.action);
  }
}

function fmt(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function logLine(text) {
  const stamp = new Date().toLocaleTimeString();
  el.scoreJson.textContent += `[${stamp}] ${text}\n`;
  el.scoreJson.scrollTop = el.scoreJson.scrollHeight;
}

function logCheck(name, analysis) {
  const check = analysis.check || {};
  const risk = check.risk != null ? ` · риск ${fmt(check.risk)}` : "";
  const reason = check.reason ? ` — ${reasonRu(check.reason)}` : "";
  logLine(`${name}: ${statusRu(analysis.status)}${risk}${reason}`);
}

// Countdown in the stage header while MediaRecorder runs, so a 5-9s silent
// recording does not read as a frozen page. Returns a stop function.
function startCountdown(totalMs, label) {
  const startedAt = performance.now();
  const render = () => {
    const left = Math.max(0, totalMs - (performance.now() - startedAt));
    el.stageValue.textContent = `${label} ${Math.ceil(left / 1000)}с`;
  };
  render();
  const timer = window.setInterval(render, 250);
  return () => window.clearInterval(timer);
}

async function recordWithCountdown(durationMs, label) {
  const stop = startCountdown(durationMs, label);
  el.stage.classList.add("recording");
  try {
    return await recordVideoBlob(durationMs);
  } finally {
    stop();
    el.stage.classList.remove("recording");
  }
}

// Short "get ready" pause before a step suddenly demands something of the
// user (full-screen flashing, a one-shot phrase).
async function prepPause(label, hint, ms) {
  el.stepHint.textContent = hint;
  const stop = startCountdown(ms, label);
  try {
    await sleep(ms);
  } finally {
    stop();
  }
}

// Spinner next to the stage headline while the server chews on an upload.
async function analyze(request) {
  el.stage.classList.add("analyzing");
  try {
    return await request;
  } finally {
    el.stage.classList.remove("analyzing");
  }
}

// Count-up in the stage header for waits with no known duration (the first
// rPPG call can sit out a ~1 min model warmup). Returns a stop function.
function startElapsed(label) {
  const startedAt = performance.now();
  const render = () => {
    el.stageValue.textContent = `${label} ${Math.round((performance.now() - startedAt) / 1000)}с`;
  };
  render();
  const timer = window.setInterval(render, 1000);
  return () => window.clearInterval(timer);
}

function displayValue(step) {
  if (step.id === "gesture" && state.session) {
    return getStep("gesture").prompt;
  }
  if (step.id === "audio" && state.session) {
    return getStep("audio_phrase").prompt;
  }
  return step.value;
}

function displayHint(step) {
  if (step.id === "audio") {
    return "Произнесите фразу вслух — сервер распознает и проверит её.";
  }
  return step.hint;
}

function advance() {
  state.stepIndex = Math.min(state.stepIndex + 1, FLOW.length - 1);
  renderStep();
}

// Release the camera and its sampling timers: without this the camera light
// stays on after reset and every restart orphans another live stream.
function stopCamera() {
  if (state.faceTimer) {
    window.clearInterval(state.faceTimer);
    state.faceTimer = null;
  }
  if (state.pulseTimer) {
    window.clearInterval(state.pulseTimer);
    state.pulseTimer = null;
  }
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  el.camera.srcObject = null;
  el.camera.classList.remove("live");
  el.faceGuide.classList.remove("visible");
  state.facePresent = null;
  state.faceConfidence = null;
}

function resetEvidence() {
  stopCamera();
  state.stepIndex = 0;
  state.scored = false;
  state.skipped = new Set();
  state.stepStatus = {};
  state.expectedLuma = [];
  state.observedLuma = [];
  state.pulse = null;
  state.audio = null;
  state.pulseSamples = [];
  state.facePresent = null;
  state.faceConfidence = null;
  state.gestureAttempt = null;
  state.serviceEvidence = {};
  el.decision.className = "decision";
  el.decision.dataset.decision = "";
  el.decision.textContent = "не оценено";
  el.riskLine.textContent = "";
  el.checksBreakdown.innerHTML = "";
  el.scoreJson.textContent = "";
  el.stage.style.backgroundColor = "#202020";
  renderFaceState();
}

// Errors surface in the stage overlay instead of a browser alert: the flow
// stays on the failed step, so the same button doubles as the retry.
function showInlineError(error) {
  const message = error?.message || String(error);
  el.stageValue.textContent = "ОШИБКА";
  el.stepHint.textContent = `${message} — нажмите кнопку ещё раз, чтобы повторить.`;
  logLine(`ошибка: ${message}`);
}

async function startSession() {
  let failure = null;
  try {
    el.primaryAction.disabled = true;
    el.resetFlow.disabled = true;
    resetEvidence();
    state.session = null;
    setStatus("создание сессии");
    state.session = await requestJson("/v1/sessions", {
      method: "POST",
      body: JSON.stringify({
        uid: el.uid.value,
        check_id: el.checkId.value,
        scenario: "frontend_sequential_challenge",
      }),
    });
    setStatus("сессия готова");
  } catch (error) {
    failure = error;
    setStatus("ошибка сессии");
  } finally {
    el.resetFlow.disabled = false;
    renderStep();
    if (failure) showInlineError(failure);
  }
}

el.primaryAction.addEventListener("click", async () => {
  if (!state.session || (currentFlowStep().id === "score" && state.scored)) {
    return startSession();
  }
  const step = currentFlowStep();
  let failure = null;
  try {
    // A reset mid-step would mix the in-flight step's evidence into a fresh
    // session — lock the flow controls until the step settles.
    el.primaryAction.disabled = true;
    el.resetFlow.disabled = true;
    el.skipStep.disabled = true;
    await runStep(step.id);
    if (step.id !== "score") {
      advance();
    }
  } catch (error) {
    failure = error;
    setStatus(`ошибка: ${step.id}`);
  } finally {
    el.resetFlow.disabled = false;
    renderStep();
    if (failure) showInlineError(failure);
  }
});

el.skipStep.addEventListener("click", () => {
  if (!TEST_SKIP_ENABLED || !state.session) return;
  const step = currentFlowStep();
  state.skipped.add(step.id);
  applySkipEvidence(step.id);
  setStatus(`пропущено: ${step.id}`);
  advance();
});

el.resetFlow.addEventListener("click", () => {
  resetEvidence();
  state.session = null;
  renderStep();
  setStatus("ожидание");
});

async function runStep(id) {
  if (id === "camera") return startCamera();
  if (id === "active_light") return runLight();
  if (id === "gesture") return confirmGesture();
  if (id === "rppg") return samplePulse();
  if (id === "audio") return recordAudio();
  if (id === "score") return submitEvidence();
}

async function startCamera() {
  setStatus("запрос доступа к камере");
  // The XGB forensic classifier needs the face near its 512px training crop;
  // the browser default 640x480 leaves faces ~200px and its verdict gated off.
  state.stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  el.camera.srcObject = state.stream;
  await waitForVideo();
  el.camera.classList.add("live");
  el.faceGuide.classList.add("visible");
  await updateFacePresence();
  renderFaceState();
  startFaceWatch();
  startPulseCollection();
  logLine(`камера: поток запущен (${el.camera.videoWidth}x${el.camera.videoHeight})`);
  setStatus("камера готова");
}

// FaceDetector is available only in some Chromium builds; when it is absent
// the oval stays neutral and the metric says so instead of guessing.
function startFaceWatch() {
  if (state.faceTimer || !("FaceDetector" in window)) {
    renderFaceState();
    return;
  }
  state.faceTimer = window.setInterval(async () => {
    if (!state.stream || state.suppressPulseCollection) return;
    await updateFacePresence();
    renderFaceState();
  }, 2000);
}

function renderFaceState() {
  el.faceOval.classList.toggle("ok", state.facePresent === true);
  el.faceOval.classList.toggle("no-face", state.facePresent === false);
  if (state.stream && state.facePresent === true) {
    el.guideHint.textContent = "Лицо обнаружено — сохраняйте положение";
  } else if (state.stream && state.facePresent === false) {
    el.guideHint.textContent = "Лицо не найдено — переместитесь в овал";
  } else {
    el.guideHint.textContent = "Совместите лицо с овалом";
  }
}

async function runLight() {
  const step = getStep("active_light");
  // A positively absent face makes the flash dance pointless: skip the
  // capture and upload and let the server fail the check instantly.
  // (facePresent === null means the browser has no FaceDetector — proceed.)
  // One fastMode sample false-negatives routinely (motion blur, turned
  // head), so only fast-fail on an absence confirmed across three samples.
  await updateFacePresence();
  for (let retry = 0; retry < 2 && state.facePresent === false; retry += 1) {
    await sleep(400);
    await updateFacePresence();
  }
  if (state.facePresent === false) {
    const analysis = await requestJson("/v1/services/active-light/analyze", {
      method: "POST",
      body: JSON.stringify({ face_present: false, face_confidence: state.faceConfidence ?? 0 }),
    });
    state.serviceEvidence.active_light = analysis.evidence;
    state.stepStatus.active_light = analysis.status;
    logCheck("active_light", analysis);
    setStatus(`свет: ${statusRu(analysis.status)}`);
    return;
  }
  await prepPause("ВСПЫШКИ ЧЕРЕЗ", "Сейчас экран будет мигать цветными вспышками. Держите лицо в овале.", 2500);
  if (Array.isArray(step.payload.face_flash_pairs) && step.payload.face_flash_pairs.length) {
    return runFaceFlashLight(step.payload.face_flash_pairs);
  }
  const sequence = step.payload.luma_sequence;
  state.expectedLuma = [...sequence];
  state.observedLuma = [];
  el.currentStep.textContent = "Проверка активным светом";
  setStatus("активный свет");

  await enterFullscreenIfPossible();
  el.flashFullscreen.classList.add("visible");
  state.suppressPulseCollection = true;
  try {
    for (const value of sequence) {
      const color = value > 127 ? "#ffffff" : "#000000";
      el.flashFullscreen.style.backgroundColor = color;
      el.stage.style.backgroundColor = color;
      el.stageValue.textContent = value > 127 ? "БЕЛЫЙ" : "ЧЁРНЫЙ";
      await sleep(LIGHT_SETTLE_MS);
      // Neutral fallback: if the camera frame is unavailable the sample must NOT
      // default to the expected value, or the check passes without a camera.
      state.observedLuma.push(await sampleStableCameraLuma(128));
    }
  } finally {
    el.flashFullscreen.classList.remove("visible");
    state.suppressPulseCollection = false;
    el.stage.style.backgroundColor = "#202020";
    await exitFullscreenIfOwned();
  }

  const analysis = await requestJson("/v1/services/active-light/analyze", {
    method: "POST",
    body: JSON.stringify({
      expected_luma: state.expectedLuma,
      observed_face_luma: state.observedLuma,
      face_present: state.facePresent,
      face_confidence: state.faceConfidence,
    }),
  });
  state.serviceEvidence.active_light = analysis.evidence;
  state.stepStatus.active_light = analysis.status;
  logCheck("active_light", analysis);
  setStatus("свет записан");
}

async function runFaceFlashLight(pairs) {
  el.currentStep.textContent = "Проверка вспышками света";
  setStatus("вспышки света");

  let analysis = await captureAndAnalyzeFlashPairs(pairs);
  if (analysis.status !== "passed") {
    logLine(`активный свет: ${statusRu(analysis.status)} — повторная попытка`);
    setStatus("свет: повторная попытка");
    analysis = await captureAndAnalyzeFlashPairs(pairs);
  }

  state.serviceEvidence.active_light = analysis.evidence;
  state.stepStatus.active_light = analysis.status;
  state.expectedLuma = pairs.map((pair) => pair.lighting.lighting_rgb?.[0] ?? 255);
  state.observedLuma = new Array(pairs.length).fill(0);
  logCheck("active_light", analysis);
  setStatus(`свет: ${statusRu(analysis.status)}`);
}

async function captureAndAnalyzeFlashPairs(pairs) {
  const manifestPairs = [];
  const form = new FormData();

  await enterFullscreenIfPossible();
  el.flashFullscreen.classList.add("visible");
  state.suppressPulseCollection = true;
  try {
    // Warmup phase: let auto-exposure settle into the flashing regime before
    // the first scored capture. This frame is shown but never uploaded.
    renderFaceFlashFrame(pairs[0].background);
    el.stageValue.textContent = "ПОДГОТОВКА";
    await settleFlashPhase();

    for (let index = 0; index < pairs.length; index += 1) {
      const pair = pairs[index];
      renderFaceFlashFrame(pair.background);
      el.stageValue.textContent = `ФОН ${index + 1}/${pairs.length}`;
      await settleFlashPhase();
      const backgroundFile = `active_light_bg_${index}.png`;
      // The verifier crops faces to 256px — full-res 720p PNGs only bloat the
      // upload (16 files, tens of MB through a tunnel); 640-wide is plenty.
      form.append("files", await captureCameraPngBlob(640), backgroundFile);

      renderFaceFlashFrame(pair.lighting);
      el.stageValue.textContent = `СВЕТ ${index + 1}/${pairs.length}`;
      await settleFlashPhase();
      const lightingFile = `active_light_light_${index}.png`;
      form.append("files", await captureCameraPngBlob(640), lightingFile);

      manifestPairs.push({
        background_file: backgroundFile,
        lighting_file: lightingFile,
        background_challenge: pair.background,
        lighting_challenge: pair.lighting,
      });
    }
  } finally {
    el.flashFullscreen.classList.remove("visible");
    state.suppressPulseCollection = false;
    el.flashFullscreen.style.backgroundImage = "";
    el.stage.style.backgroundImage = "";
    el.stage.style.backgroundColor = "#202020";
    await exitFullscreenIfOwned();
  }

  form.append("manifest", JSON.stringify({ pairs: manifestPairs }));
  el.stageValue.textContent = "анализ…";
  return analyze(requestForm("/v1/services/active-light/analyze-frame-pairs", form));
}

// The camera pipeline lags the screen by several frames, so a capture right
// after a flash change often still shows the previous light. Wait out the
// pacing floor, then require two frames actually sensed under the new light.
// The frame wait is capped: dark phases push webcams into long exposure at a
// few fps (and fullscreen can starve rVFC), which would stretch every phase
// and the whole challenge with it.
async function settleFlashPhase() {
  await sleep(FLASH_PHASE_MS);
  await Promise.race([nextCameraFrames(2), sleep(FLASH_FRAME_WAIT_MAX_MS)]);
}

function nextCameraFrames(count) {
  return new Promise((resolve) => {
    if (typeof el.camera.requestVideoFrameCallback !== "function") {
      window.setTimeout(resolve, count * 67); // ~count frames at 30fps
      return;
    }
    let remaining = count;
    const tick = () => {
      remaining -= 1;
      if (remaining <= 0) resolve();
      else el.camera.requestVideoFrameCallback(tick);
    };
    el.camera.requestVideoFrameCallback(tick);
  });
}

function renderFaceFlashFrame(challenge) {
  const bg = rgbCss(challenge.background_rgb);
  let image = "";
  if (challenge.kind === "lighting" && challenge.lighting_rgb) {
    const top = (challenge.stripe_top / challenge.height) * 100;
    const bottom = (challenge.stripe_bottom / challenge.height) * 100;
    const light = rgbCss(challenge.lighting_rgb);
    image = `linear-gradient(to bottom, ${bg} 0%, ${bg} ${top}%, ${light} ${top}%, ${light} ${bottom}%, ${bg} ${bottom}%, ${bg} 100%)`;
  }
  el.flashFullscreen.style.backgroundColor = bg;
  el.flashFullscreen.style.backgroundImage = image;
  el.stage.style.backgroundColor = bg;
  el.stage.style.backgroundImage = image;
}

function rgbCss(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

async function confirmGesture() {
  const gesture = getStep("gesture");
  setStatus("запись жеста");
  const blob = await recordWithCountdown(gesture.duration_ms || 5000, "ЗАПИСЬ");
  const form = new FormData();
  form.append("file", blob, "gesture.webm");
  form.append("expected_action", gesture.payload.expected_action);
  if (state.facePresent !== null) form.append("face_present", String(state.facePresent));
  el.stageValue.textContent = "анализ…";
  setStatus("жест: анализ");
  const analysis = await analyze(requestForm("/v1/services/gesture/analyze-video", form));
  state.serviceEvidence.gesture = analysis.evidence;
  state.gestureAttempt = analysis.evidence;
  state.stepStatus.gesture = analysis.status;
  logCheck("gesture", analysis);
  setStatus(`жест: ${statusRu(analysis.status)}`);
}

async function samplePulse() {
  setStatus("запись видео rPPG");
  try {
    // 18s instead of the original 9: rPPG needs a long stable face window,
    // and short clips were the main source of low-SQI "unknown" verdicts.
    const blob = await recordWithCountdown(18000, "ПУЛЬС");
    const form = new FormData();
    form.append("file", blob, "rppg.webm");
    if (state.facePresent !== null) form.append("face_present", String(state.facePresent));
    if (state.faceConfidence !== null) form.append("face_confidence", String(state.faceConfidence));
    setStatus("пульс: анализ (первый запуск до ~1 мин)");
    const stopElapsed = startElapsed("АНАЛИЗ");
    let analysis;
    try {
      analysis = await analyze(requestForm("/v1/services/rppg/analyze-video", form));
    } finally {
      stopElapsed();
    }
    state.serviceEvidence.rppg = analysis.evidence;
    state.pulse = {
      bpm: analysis.evidence.bpm ?? null,
      signal_quality: analysis.evidence.signal_quality ?? null,
    };
    state.stepStatus.rppg = analysis.status;
    logCheck("rppg", analysis);
    setStatus(`пульс: ${statusRu(analysis.status)}`);
    return;
  } catch (_error) {
    logLine("rppg: загрузка видео не удалась, используются образцы яркости");
    setStatus("видео rPPG не удалось, используются образцы");
  }

  for (let index = 0; index < 80; index += 1) {
    el.stageValue.textContent = `${index + 1}/80`;
    await sleep(100);
  }
  state.pulse = estimatePulse(getPulseValues());
  const analysis = await requestJson("/v1/services/rppg/analyze", {
    method: "POST",
    body: JSON.stringify({
      samples: getPulseValues(),
      sample_rate_hz: 10,
      window_seconds: 4,
      face_present: state.facePresent,
      face_confidence: state.faceConfidence,
    }),
  });
  state.serviceEvidence.rppg = analysis.evidence;
  state.stepStatus.rppg = analysis.status;
  logCheck("rppg", analysis);
  setStatus("пульс замерен");
}

// Level check on arbitrary speech BEFORE any phrase is disclosed: re-tries at
// this stage cost nothing security-wise, because no challenge is in the open
// yet. Advisory only — a stubbornly quiet mic logs a warning, never a dead end.
async function micPreflight() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (_error) {
    return; // recording will fail later with its own message
  }
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    try {
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const data = new Float32Array(analyser.fftSize);
      for (let attempt = 0; attempt < MIC_PREFLIGHT_TRIES; attempt += 1) {
        setStatus("проверка микрофона");
        el.stepHint.textContent = "Скажите что-нибудь вслух — проверяем уровень микрофона.";
        const stop = startCountdown(MIC_PREFLIGHT_MS, "СКАЖИТЕ ЧТО-НИБУДЬ");
        let peak = 0;
        const startedAt = performance.now();
        try {
          while (performance.now() - startedAt < MIC_PREFLIGHT_MS) {
            await sleep(100);
            analyser.getFloatTimeDomainData(data);
            let sum = 0;
            for (let i = 0; i < data.length; i += 1) sum += data[i] * data[i];
            peak = Math.max(peak, Math.sqrt(sum / data.length));
          }
        } finally {
          stop();
        }
        if (peak >= MIC_MIN_RMS) {
          logLine(`микрофон: уровень в норме (RMS ${peak.toFixed(3)})`);
          return;
        }
        logLine(`микрофон: очень тихо (RMS ${peak.toFixed(3)}), попытка ${attempt + 1}/${MIC_PREFLIGHT_TRIES}`);
        el.stageValue.textContent = "ОЧЕНЬ ТИХО";
        el.stepHint.textContent = "Говорите громче или поднесите микрофон ближе.";
        await sleep(1200);
      }
      logLine("микрофон: уровень так и не поднялся — продолжаем, проверка может не пройти");
    } finally {
      await ctx.close();
    }
  } catch (_error) {
    // No AudioContext/analyser — skip the preflight rather than block the flow.
  } finally {
    stream.getTracks().forEach((track) => track.stop());
  }
}

async function recordAudio() {
  const audioStep = getStep("audio_phrase");
  await micPreflight();
  await prepPause(
    "ФРАЗА ЧЕРЕЗ",
    "Произнесите фразу сразу, как она появится. Попытка одна — говорите чётко.",
    3000,
  );

  // One phrase, one recording, no retries: every extra attempt re-opens the
  // window between phrase disclosure and submission — exactly the time a
  // fraudster needs to synthesize the phrase with a cloned voice.
  let issue;
  try {
    issue = await requestJson(`/v1/sessions/${state.session.session_id}/audio/phrase`, { method: "POST" });
  } catch (error) {
    logLine(`аудио: фраза не выдана (${error.message})`);
    setStatus("аудио записано");
    return;
  }
  audioStep.prompt = issue.phrase;
  audioStep.payload.phrase = issue.phrase;
  el.stageValue.textContent = issue.phrase;
  setStatus("запись аудио");

  // Only genuine capture failures fall to the browser_recording_failed
  // path; upload/server errors propagate to the step handler so the user
  // sees the real cause.
  let blob;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const started = performance.now();
    const stopCountdown = startCountdown(audioStep.duration_ms || 6000, "ГОВОРИТЕ");
    el.stepHint.textContent = `Произнесите: «${issue.phrase}»`;
    el.stage.classList.add("recording");
    try {
      blob = await recordStreamBlob(stream, audioStep.duration_ms || 6000);
    } finally {
      el.stage.classList.remove("recording");
      stopCountdown();
      stream.getTracks().forEach((track) => track.stop());
    }
    state.audio = {
      duration_seconds: (performance.now() - started) / 1000,
    };
  } catch (_error) {
    state.audio = { duration_seconds: 3.0 };
    state.serviceEvidence.audio = {
      phrase_expected: audioStep.payload.phrase,
      duration_seconds: 3.0,
      detector: "browser_recording_failed",
    };
    state.stepStatus.audio = "unknown";
    logLine("аудио: запись в браузере не удалась");
    setStatus("аудио записано");
    return;
  }

  const form = new FormData();
  form.append("file", blob, "audio.webm");
  el.stageValue.textContent = "анализ…";
  setStatus("аудио: анализ");
  const analysis = await analyze(requestForm(`/v1/sessions/${state.session.session_id}/audio/analyze`, form));

  state.serviceEvidence.audio = analysis.evidence;
  state.stepStatus.audio = analysis.status;
  logCheck("audio", analysis);
  if (analysis.evidence.phrase_transcribed != null) {
    logLine(`аудио: сервер распознал "${analysis.evidence.phrase_transcribed}"`);
  }
  setStatus("аудио записано");
}

async function analyzeClassifier() {
  if (!state.stream) return;
  setStatus("запись клипа классификатора");
  const blob = await recordWithCountdown(2500, "ЗАПИСЬ");
  el.stageValue.textContent = "анализ…";
  setStatus("классификатор: анализ");
  const form = new FormData();
  form.append("file", blob, "classifier.webm");
  if (state.facePresent !== null) form.append("face_present", String(state.facePresent));
  if (state.faceConfidence !== null) form.append("face_confidence", String(state.faceConfidence));
  const analysis = await analyze(requestForm("/v1/services/classifier/analyze-video", form));
  state.serviceEvidence.classifier = analysis.evidence;
  logCheck("classifier", analysis);
  setStatus(`классификатор: ${statusRu(analysis.status)}`);
}

async function submitEvidence() {
  await updateFacePresence();
  if (!state.serviceEvidence.classifier && state.stream) {
    try {
      await analyzeClassifier();
    } catch (_error) {
      state.serviceEvidence.classifier = {
        face_present: state.facePresent,
        face_confidence: state.faceConfidence,
      };
    }
  }
  const gesture = getStep("gesture");
  const audio = getStep("audio_phrase");
  const observed = state.observedLuma.length ? state.observedLuma : state.expectedLuma;
  const pulse = state.pulse || { bpm: null, signal_quality: null };
  const payload = {
    uid: state.session.uid,
    check_id: state.session.check_id,
    evidence: {
      classifier: state.serviceEvidence.classifier || {
        frame_count: observed.length,
        face_present: state.facePresent,
        face_confidence: state.faceConfidence,
      },
      active_light: state.serviceEvidence.active_light || {
        skipped: state.skipped.has("active_light"),
        expected_luma: state.expectedLuma,
        observed_face_luma: observed,
        face_present: state.facePresent,
        face_confidence: state.faceConfidence,
      },
      rppg: state.serviceEvidence.rppg || {
        skipped: state.skipped.has("rppg"),
        face_present: state.facePresent,
        face_confidence: state.faceConfidence,
        bpm: state.skipped.has("rppg") ? null : pulse.bpm,
        signal_quality: state.skipped.has("rppg") ? null : pulse.signal_quality,
        samples: getPulseValues(),
        sample_rate_hz: 10,
        window_seconds: 4,
      },
      gesture: state.serviceEvidence.gesture || {
        skipped: state.skipped.has("gesture"),
        expected_action: gesture.payload.expected_action,
        observed_action: state.gestureAttempt?.observed_action,
        confidence: state.gestureAttempt?.confidence,
        detector: state.gestureAttempt?.detector,
        face_present: state.facePresent,
      },
      audio: state.serviceEvidence.audio || {
        skipped: state.skipped.has("audio"),
        phrase_expected: audio.payload.phrase,
        duration_seconds: state.audio?.duration_seconds || 0,
      },
    },
  };

  setStatus("оценка");
  const result = await requestJson(`/v1/sessions/${state.session.session_id}/evidence`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.scored = true;
  // The check is over — release the camera right away instead of waiting
  // for a manual reset.
  stopCamera();
  el.decision.className = `decision ${result.decision}`;
  el.decision.dataset.decision = result.decision;
  el.decision.textContent = statusRu(result.decision);
  el.riskLine.textContent = `риск ${fmt(result.risk_score)} · уверенность ${fmt(result.confidence)}`;
  renderChecksBreakdown(result);
  logLine(`оценка: ${statusRu(result.decision)} (риск ${fmt(result.risk_score)})`);
  el.scoreJson.textContent += `\n${JSON.stringify(result, null, 2)}\n`;
  el.scoreJson.scrollTop = el.scoreJson.scrollHeight;
  setStatus("оценено");
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  // innerHTML escapes <>& but not quotes, and the result lands in title="…".
  return div.innerHTML.replace(/"/g, "&quot;");
}

function renderChecksBreakdown(result) {
  const rows = (result.checks || [])
    .map((check) => {
      const width = Math.round(check.risk * 100);
      return `<div class="check-row ${check.status}" title="${escapeHtml(reasonRu(check.reason))}">
        <span class="check-name">${escapeHtml(checkNameRu(check.name))}</span>
        <span class="check-status">${escapeHtml(statusRu(check.status))}</span>
        <span class="check-riskbar"><i style="width:${width}%"></i></span>
        <span class="check-risk">${fmt(check.risk)}</span>
      </div>`;
    })
    .join("");
  const factors = result.factors?.length
    ? `<p class="factors">${result.factors.map((factor) => escapeHtml(factorRu(factor))).join(" · ")}</p>`
    : "";
  el.checksBreakdown.innerHTML = rows + factors;
}

function applySkipEvidence(id) {
  state.stepStatus[id] = "skipped";
  logLine(`${id}: пропущено (тестовый режим)`);
  if (id === "camera") return;
  if (id === "active_light") {
    const step = getStep("active_light");
    state.expectedLuma = [...step.payload.luma_sequence];
    state.observedLuma = [...step.payload.luma_sequence];
    state.serviceEvidence.active_light = { skipped: true };
  }
  if (id === "gesture") {
    state.gestureAttempt = { detector: "skipped", observed_action: null, confidence: 0 };
    state.serviceEvidence.gesture = { skipped: true };
  }
  if (id === "rppg") {
    state.pulse = { bpm: null, signal_quality: null };
    state.serviceEvidence.rppg = { skipped: true };
  }
  if (id === "audio") {
    state.audio = { duration_seconds: 0 };
    state.serviceEvidence.audio = { skipped: true };
  }
}

function estimatePulse(samples) {
  if (!samples.length) return { bpm: null, signal_quality: null };
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const centered = samples.map((value) => value - mean);
  const peak = Math.max(...centered) - Math.min(...centered);
  const noise = centered.slice(1).reduce((sum, value, index) => sum + Math.abs(value - centered[index]), 0) / Math.max(1, centered.length - 1);
  const quality = Math.max(0, Math.min(1, peak / (peak + noise + 1e-6)));
  return { bpm: null, signal_quality: quality };
}

function startPulseCollection() {
  if (state.pulseTimer) return;
  state.pulseTimer = window.setInterval(() => {
    if (!state.stream || el.camera.readyState < 2) return;
    if (state.suppressPulseCollection) return;
    state.pulseSamples.push({ time: performance.now(), value: sampleCameraLuma(128) });
    const cutoff = performance.now() - 120000;
    while (state.pulseSamples.length && state.pulseSamples[0].time < cutoff) {
      state.pulseSamples.shift();
    }
  }, 100);
}

function getPulseValues() {
  return state.pulseSamples.map((sample) => sample.value);
}

async function recordVideoBlob(durationMs) {
  if (!state.stream) {
    throw new Error("Для записи жеста требуется поток камеры");
  }
  return recordStreamBlob(state.stream, durationMs);
}

async function captureCameraPngBlob(maxWidth) {
  if (!state.stream || el.camera.readyState < 2) {
    throw new Error("Для захвата кадра требуется поток камеры");
  }
  const canvas = document.createElement("canvas");
  let width = el.camera.videoWidth || 640;
  let height = el.camera.videoHeight || 480;
  if (maxWidth && width > maxWidth) {
    height = Math.round(height * (maxWidth / width));
    width = maxWidth;
  }
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(el.camera, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Не удалось закодировать кадр камеры"));
    }, "image/png");
  });
}

async function recordStreamBlob(stream, durationMs) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let recorder;
    try {
      // Default ~2.5 Mbps VP8 smears the high-frequency detail the forensic
      // video classifier scores; ask for more (ignored on audio-only streams).
      recorder = new MediaRecorder(stream, { videoBitsPerSecond: 6_000_000 });
    } catch (error) {
      reject(error);
      return;
    }
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size) chunks.push(event.data);
    };
    recorder.onerror = () => reject(recorder.error || new Error("Сбой MediaRecorder"));
    recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
    recorder.start();
    window.setTimeout(() => {
      if (recorder.state !== "inactive") recorder.stop();
    }, durationMs);
  });
}

async function waitForVideo() {
  for (let index = 0; index < 20; index += 1) {
    if (el.camera.readyState >= 2 && el.camera.videoWidth > 0) return;
    await sleep(100);
  }
}

async function updateFacePresence() {
  if (!("FaceDetector" in window) || el.camera.readyState < 2) {
    state.facePresent = null;
    state.faceConfidence = null;
    return;
  }
  try {
    const detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
    const faces = await detector.detect(el.camera);
    state.facePresent = faces.length > 0;
    state.faceConfidence = faces.length > 0 ? 0.85 : 0.0;
  } catch (_error) {
    state.facePresent = null;
    state.faceConfidence = null;
  }
}

function sampleCameraLuma(fallbackValue) {
  if (!state.stream || el.camera.readyState < 2) {
    return fallbackValue;
  }

  const canvas = document.createElement("canvas");
  const width = 96;
  const height = 72;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(el.camera, 0, 0, width, height);
  const cropX = Math.floor(width * 0.25);
  const cropY = Math.floor(height * 0.2);
  const cropW = Math.floor(width * 0.5);
  const cropH = Math.floor(height * 0.55);
  const data = ctx.getImageData(cropX, cropY, cropW, cropH).data;
  let total = 0;
  for (let index = 0; index < data.length; index += 4) {
    total += 0.2126 * data[index] + 0.7152 * data[index + 1] + 0.0722 * data[index + 2];
  }
  return total / (data.length / 4);
}

async function sampleStableCameraLuma(fallbackValue) {
  const samples = [];
  for (let index = 0; index < LIGHT_SAMPLE_COUNT; index += 1) {
    samples.push(sampleCameraLuma(fallbackValue));
    await sleep(LIGHT_SAMPLE_INTERVAL_MS);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

async function enterFullscreenIfPossible() {
  if (!document.fullscreenEnabled || document.fullscreenElement) return;
  try {
    await document.documentElement.requestFullscreen();
    state.ownsFullscreen = true;
  } catch (_error) {
    state.ownsFullscreen = false;
  }
}

async function exitFullscreenIfOwned() {
  if (!state.ownsFullscreen || !document.fullscreenElement) return;
  try {
    await document.exitFullscreen();
  } catch (_error) {
    return;
  } finally {
    state.ownsFullscreen = false;
  }
}

FLOW.forEach(() => el.progressDots.appendChild(document.createElement("i")));
renderStep();
