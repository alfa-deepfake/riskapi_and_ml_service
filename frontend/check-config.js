// Tuning constants and the check step list. Loaded before app.js so these
// globals are in scope for the flow logic; config.js (window.APP_CONFIG)
// must load first.
const appConfig = window.APP_CONFIG || {};
const TEST_SKIP_ENABLED = Boolean(appConfig.enableTestSkip);
const LIGHT_SETTLE_MS = Number(appConfig.lightSettleMs || 180);
// Flash pacing floor: ≥250ms per phase keeps the strobe at ≤2 flashes/s —
// safely under the WCAG 2.3.1 photosensitivity limit of 3 flashes/s — and
// gives camera auto-exposure time to react to each change.
const FLASH_PHASE_MS = Number(appConfig.flashPhaseMs || 250);
const FLASH_FRAME_WAIT_MAX_MS = Number(appConfig.flashFrameWaitMaxMs || 400);
// Cap for waiting on a real screen repaint after a flash color change. rAF
// pauses on a backgrounded tab, so this timer keeps the loop from hanging
// there; in the foreground the double-rAF resolves first (~1 frame).
const FLASH_PRESENT_WAIT_MS = Number(appConfig.flashPresentWaitMs || 300);
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
