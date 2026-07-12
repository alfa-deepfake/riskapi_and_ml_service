const appConfig = window.APP_CONFIG || {};
const TEST_SKIP_ENABLED = Boolean(appConfig.enableTestSkip);
const LIGHT_SETTLE_MS = Number(appConfig.lightSettleMs || 180);
const LIGHT_SAMPLE_COUNT = Number(appConfig.lightSampleCount || 4);
const LIGHT_SAMPLE_INTERVAL_MS = Number(appConfig.lightSampleIntervalMs || 70);

const FLOW = [
  {
    id: "camera",
    title: "Camera check",
    value: "camera",
    hint: "Allow camera access. The stream is used by light and pulse checks.",
    action: "Start camera",
  },
  {
    id: "active_light",
    title: "Active light challenge",
    value: "black / white",
    hint: "Keep your face in frame while the screen flashes black and white.",
    action: "Run flashes",
  },
  {
    id: "gesture",
    title: "Gesture challenge",
    value: "gesture",
    hint: "Perform the gesture shown by the challenge.",
    action: "Confirm gesture",
  },
  {
    id: "rppg",
    title: "Pulse check",
    value: "pulse",
    hint: "Hold still for a short rPPG sampling window.",
    action: "Sample pulse",
  },
  {
    id: "audio",
    title: "Audio phrase challenge",
    value: "phrase",
    hint: "Say the generated phrase. For MVP the transcript field is editable.",
    action: "Record audio",
  },
  {
    id: "score",
    title: "Risk score",
    value: "score",
    hint: "Submit collected evidence to ML service and risk-api callback.",
    action: "Submit score",
  },
];

const state = {
  session: null,
  stream: null,
  pulseTimer: null,
  pulseSamples: [],
  suppressPulseCollection: false,
  facePresent: null,
  faceConfidence: null,
  gestureAttempt: null,
  serviceEvidence: {},
  stepIndex: 0,
  skipped: new Set(),
  stepStatus: {},
  expectedLuma: [],
  observedLuma: [],
  gestureDone: false,
  pulse: null,
  audio: null,
};

const el = {
  apiUrl: document.querySelector("#apiUrl"),
  uid: document.querySelector("#uid"),
  checkId: document.querySelector("#checkId"),
  status: document.querySelector("#status"),
  startVerification: document.querySelector("#startVerification"),
  primaryAction: document.querySelector("#primaryAction"),
  skipStep: document.querySelector("#skipStep"),
  resetFlow: document.querySelector("#resetFlow"),
  stage: document.querySelector("#challengeStage"),
  flashFullscreen: document.querySelector("#flashFullscreen"),
  camera: document.querySelector("#camera"),
  currentStep: document.querySelector("#currentStep"),
  stageValue: document.querySelector("#stageValue"),
  stepHint: document.querySelector("#stepHint"),
  lightMetric: document.querySelector("#lightMetric"),
  pulseMetric: document.querySelector("#pulseMetric"),
  gestureMetric: document.querySelector("#gestureMetric"),
  audioMetric: document.querySelector("#audioMetric"),
  phraseInput: document.querySelector("#phraseInput"),
  decision: document.querySelector("#decision"),
  scoreJson: document.querySelector("#scoreJson"),
};

el.apiUrl.value = appConfig.mlApiUrl || "http://localhost:8100";
el.uid.value = `user-${Math.random().toString(16).slice(2, 8)}`;
el.checkId.value = `check-${Date.now()}`;
el.skipStep.style.display = TEST_SKIP_ENABLED ? "block" : "none";

function setStatus(value) {
  el.status.textContent = value;
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

async function requestJson(path, options = {}) {
  const response = await fetch(api(path), {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

async function requestForm(path, form) {
  const response = await fetch(api(path), {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

function renderStep() {
  const step = currentFlowStep();
  if (!state.session) {
    el.primaryAction.disabled = true;
    el.skipStep.disabled = true;
    el.currentStep.textContent = "Create a session to start";
    el.stageValue.textContent = "--";
    el.stepHint.textContent = "Press Start verification";
    updateTools();
    return;
  }

  el.primaryAction.disabled = false;
  el.skipStep.disabled = !TEST_SKIP_ENABLED || step.id === "score";
  el.currentStep.textContent = step.title;
  el.stageValue.textContent = displayValue(step);
  el.stepHint.textContent = displayHint(step);
  el.primaryAction.textContent = step.action;
  updateTools();
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
    return "Browser records audio for duration metadata; transcript can be edited for test flow.";
  }
  return step.hint;
}

function updateTools() {
  document.querySelectorAll(".tool").forEach((item) => {
    const id = item.dataset.tool;
    const s = toolState(id);
    item.classList.toggle("active", state.session && currentFlowStep()?.id === id);
    item.classList.toggle("done", s === "done");
    item.classList.toggle("failed", s === "failed");
    item.classList.toggle("skipped", s === "skipped");
  });
}

// A tool is only "done" (green) when its check actually PASSED. A check that ran
// but did not pass ("failed"/"unknown", e.g. no face in frame) shows "failed" —
// collecting evidence is not the same as passing the check.
function toolState(id) {
  if (state.skipped.has(id)) return "skipped";
  if (id === "camera") return state.stream ? "done" : "";
  if (id === "score") return el.decision.textContent !== "not scored" ? "done" : "";
  const status = state.stepStatus[id];
  if (!status) return "";
  return status === "passed" ? "done" : "failed";
}

function advance() {
  state.stepIndex = Math.min(state.stepIndex + 1, FLOW.length - 1);
  renderStep();
}

function resetEvidence() {
  state.stepIndex = 0;
  state.skipped = new Set();
  state.stepStatus = {};
  state.expectedLuma = [];
  state.observedLuma = [];
  state.gestureDone = false;
  state.pulse = null;
  state.audio = null;
  state.pulseSamples = [];
  state.facePresent = null;
  state.faceConfidence = null;
  state.gestureAttempt = null;
  state.serviceEvidence = {};
  el.lightMetric.textContent = "0";
  el.pulseMetric.textContent = "pending";
  el.gestureMetric.textContent = "pending";
  el.audioMetric.textContent = "pending";
  el.decision.className = "decision";
  el.decision.textContent = "not scored";
  el.scoreJson.textContent = "{}";
  el.stage.style.backgroundColor = "#202020";
}

el.startVerification.addEventListener("click", async () => {
  try {
    resetEvidence();
    setStatus("creating session");
    state.session = await requestJson("/v1/sessions", {
      method: "POST",
      body: JSON.stringify({
        uid: el.uid.value,
        check_id: el.checkId.value,
        scenario: "frontend_sequential_challenge",
      }),
    });
    el.phraseInput.value = getStep("audio_phrase").payload.phrase;
    el.gestureMetric.textContent = getStep("gesture").payload.expected_action;
    setStatus("session ready");
    renderStep();
  } catch (error) {
    setStatus("session error");
    alert(error.message);
  }
});

el.primaryAction.addEventListener("click", async () => {
  if (!state.session) return;
  const step = currentFlowStep();
  try {
    el.primaryAction.disabled = true;
    await runStep(step.id);
    if (step.id !== "score") {
      advance();
    }
  } catch (error) {
    setStatus(`${step.id} error`);
    alert(error.message);
  } finally {
    renderStep();
  }
});

el.skipStep.addEventListener("click", () => {
  if (!TEST_SKIP_ENABLED || !state.session) return;
  const step = currentFlowStep();
  state.skipped.add(step.id);
  applySkipEvidence(step.id);
  setStatus(`${step.id} skipped`);
  advance();
});

el.resetFlow.addEventListener("click", () => {
  resetEvidence();
  state.session = null;
  renderStep();
  setStatus("idle");
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
  setStatus("camera permission");
  state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  el.camera.srcObject = state.stream;
  await waitForVideo();
  await updateFacePresence();
  startPulseCollection();
  setStatus("camera ready");
}

async function runLight() {
  const step = getStep("active_light");
  if (Array.isArray(step.payload.face_flash_pairs) && step.payload.face_flash_pairs.length) {
    return runFaceFlashLight(step.payload.face_flash_pairs);
  }
  const sequence = step.payload.luma_sequence;
  state.expectedLuma = [...sequence];
  state.observedLuma = [];
  el.currentStep.textContent = "Active light challenge";
  setStatus("active light");

  await enterFullscreenIfPossible();
  el.flashFullscreen.classList.add("visible");
  state.suppressPulseCollection = true;
  try {
    for (const value of sequence) {
      const color = value > 127 ? "#ffffff" : "#000000";
      el.flashFullscreen.style.backgroundColor = color;
      el.stage.style.backgroundColor = color;
      el.stageValue.textContent = value > 127 ? "WHITE" : "BLACK";
      await sleep(LIGHT_SETTLE_MS);
      state.observedLuma.push(await sampleStableCameraLuma(value));
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
  el.lightMetric.textContent = analysis.status;
  setStatus("light captured");
}

async function runFaceFlashLight(pairs) {
  const manifestPairs = [];
  const form = new FormData();
  el.currentStep.textContent = "Face flashing challenge";
  setStatus("face flashing");

  await enterFullscreenIfPossible();
  el.flashFullscreen.classList.add("visible");
  state.suppressPulseCollection = true;
  try {
    for (let index = 0; index < pairs.length; index += 1) {
      const pair = pairs[index];
      renderFaceFlashFrame(pair.background);
      el.stageValue.textContent = `BG ${index + 1}/${pairs.length}`;
      await sleep(160);
      const backgroundFile = `active_light_bg_${index}.png`;
      form.append("files", await captureCameraPngBlob(), backgroundFile);

      renderFaceFlashFrame(pair.lighting);
      el.stageValue.textContent = `LIGHT ${index + 1}/${pairs.length}`;
      await sleep(160);
      const lightingFile = `active_light_light_${index}.png`;
      form.append("files", await captureCameraPngBlob(), lightingFile);

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
  const analysis = await requestForm("/v1/services/active-light/analyze-frame-pairs", form);
  state.serviceEvidence.active_light = analysis.evidence;
  state.stepStatus.active_light = analysis.status;
  state.expectedLuma = pairs.map((pair) => pair.lighting.lighting_rgb?.[0] ?? 255);
  state.observedLuma = new Array(pairs.length).fill(0);
  el.lightMetric.textContent = analysis.status;
  setStatus(`light ${analysis.status}`);
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
  setStatus("recording gesture");
  const blob = await recordVideoBlob(gesture.duration_ms || 5000);
  const form = new FormData();
  form.append("file", blob, "gesture.webm");
  form.append("expected_action", gesture.payload.expected_action);
  if (state.facePresent !== null) form.append("face_present", String(state.facePresent));
  const analysis = await requestForm("/v1/services/gesture/analyze-video", form);
  state.serviceEvidence.gesture = analysis.evidence;
  state.gestureAttempt = analysis.evidence;
  state.gestureDone = analysis.status === "passed";
  state.stepStatus.gesture = analysis.status;
  el.gestureMetric.textContent = analysis.status;
  setStatus(`gesture ${analysis.status}`);
}

async function samplePulse() {
  setStatus("recording rPPG video");
  try {
    const blob = await recordVideoBlob(9000);
    const form = new FormData();
    form.append("file", blob, "rppg.webm");
    if (state.facePresent !== null) form.append("face_present", String(state.facePresent));
    if (state.faceConfidence !== null) form.append("face_confidence", String(state.faceConfidence));
    const analysis = await requestForm("/v1/services/rppg/analyze-video", form);
    state.serviceEvidence.rppg = analysis.evidence;
    state.pulse = {
      bpm: analysis.evidence.bpm ?? null,
      signal_quality: analysis.evidence.signal_quality ?? null,
    };
    state.stepStatus.rppg = analysis.status;
    el.pulseMetric.textContent = analysis.status;
    setStatus(`pulse ${analysis.status}`);
    return;
  } catch (_error) {
    setStatus("rPPG video failed, using samples");
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
  el.pulseMetric.textContent = analysis.status;
  setStatus("pulse sampled");
}

async function recordAudio() {
  const audioStep = getStep("audio_phrase");
  el.stageValue.textContent = audioStep.prompt;
  setStatus("recording audio");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const started = performance.now();
    const blob = await recordStreamBlob(stream, 3000);
    stream.getTracks().forEach((track) => track.stop());
    state.audio = {
      duration_seconds: (performance.now() - started) / 1000,
    };
    const form = new FormData();
    form.append("file", blob, "audio.webm");
    form.append("phrase_expected", audioStep.payload.phrase);
    form.append("phrase_transcribed", el.phraseInput.value || "");
    const analysis = await requestForm("/v1/services/audio/analyze", form);
    state.serviceEvidence.audio = analysis.evidence;
    state.stepStatus.audio = analysis.status;
    el.audioMetric.textContent = analysis.status;
  } catch (_error) {
    state.audio = { duration_seconds: 3.0 };
    state.serviceEvidence.audio = {
      phrase_expected: audioStep.payload.phrase,
      phrase_transcribed: el.phraseInput.value || "",
      duration_seconds: 3.0,
      detector: "browser_recording_failed",
    };
    state.stepStatus.audio = "unknown";
    el.audioMetric.textContent = "unknown";
  }

  setStatus("audio captured");
}

async function analyzeClassifier() {
  if (!state.stream) return;
  setStatus("recording classifier clip");
  const blob = await recordVideoBlob(2500);
  const form = new FormData();
  form.append("file", blob, "classifier.webm");
  if (state.facePresent !== null) form.append("face_present", String(state.facePresent));
  if (state.faceConfidence !== null) form.append("face_confidence", String(state.faceConfidence));
  const analysis = await requestForm("/v1/services/classifier/analyze-video", form);
  state.serviceEvidence.classifier = analysis.evidence;
  setStatus(`classifier ${analysis.status}`);
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
        phrase_transcribed: el.phraseInput.value || "",
        duration_seconds: state.audio?.duration_seconds || 0,
      },
    },
  };

  setStatus("scoring");
  const result = await requestJson(`/v1/sessions/${state.session.session_id}/evidence`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  el.decision.className = `decision ${result.decision}`;
  el.decision.textContent = result.decision;
  el.scoreJson.textContent = JSON.stringify(result, null, 2);
  setStatus("scored");
}

function applySkipEvidence(id) {
  state.stepStatus[id] = "skipped";
  if (id === "camera") return;
  if (id === "active_light") {
    const step = getStep("active_light");
    state.expectedLuma = [...step.payload.luma_sequence];
    state.observedLuma = [...step.payload.luma_sequence];
    state.serviceEvidence.active_light = { skipped: true };
    el.lightMetric.textContent = "skipped";
  }
  if (id === "gesture") {
    state.gestureDone = false;
    state.gestureAttempt = { detector: "skipped", observed_action: null, confidence: 0 };
    state.serviceEvidence.gesture = { skipped: true };
    el.gestureMetric.textContent = "skipped";
  }
  if (id === "rppg") {
    state.pulse = { bpm: null, signal_quality: null };
    state.serviceEvidence.rppg = { skipped: true };
    el.pulseMetric.textContent = "skipped";
  }
  if (id === "audio") {
    state.audio = { duration_seconds: 0 };
    state.serviceEvidence.audio = { skipped: true };
    el.audioMetric.textContent = "skipped";
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
    throw new Error("Camera stream is required for gesture recording");
  }
  return recordStreamBlob(state.stream, durationMs);
}

async function captureCameraPngBlob() {
  if (!state.stream || el.camera.readyState < 2) {
    throw new Error("Camera stream is required for frame capture");
  }
  const canvas = document.createElement("canvas");
  canvas.width = el.camera.videoWidth || 640;
  canvas.height = el.camera.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(el.camera, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not encode camera frame"));
    }, "image/png");
  });
}

async function recordStreamBlob(stream, durationMs) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let recorder;
    try {
      recorder = new MediaRecorder(stream);
    } catch (error) {
      reject(error);
      return;
    }
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size) chunks.push(event.data);
    };
    recorder.onerror = () => reject(recorder.error || new Error("MediaRecorder failed"));
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

renderStep();
