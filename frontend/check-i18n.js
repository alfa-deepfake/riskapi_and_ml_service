// Russian display strings for server tokens (status/decision/check names/
// reason sentences). Raw English tokens stay for CSS classes and flow logic.

// The server returns English status/decision tokens (passed/failed/allow/…);
// map them to Russian for display while keeping the raw token for CSS classes
// and flow logic.
const STATUS_RU = {
  passed: "пройдено",
  failed: "не пройдено",
  unknown: "неизвестно",
  skipped: "пропущено",
  pending: "ожидание",
};

function statusRu(value) {
  return STATUS_RU[value] || value;
}

// The headline verdict is a graded deepfake-risk band derived from the numeric
// risk_score, not the raw allow/review/deny decision. A single failed check no
// longer forces a "high risk" banner — the wording (and colour) track the
// averaged risk. cls reuses the existing .decision.{allow,review,deny} styles.
// A band matches when score < max; boundaries belong to the higher band
// (0.2 → "умеренный", matching "<0.2 = низкий").
const RISK_BANDS = [
  { max: 0.2, label: "Низкий риск дипфейка", cls: "allow" },
  { max: 0.4, label: "Умеренный риск дипфейка", cls: "allow" },
  { max: 0.6, label: "Средний риск дипфейка", cls: "review" },
  { max: 0.8, label: "Высокий риск дипфейка", cls: "review" },
  { max: Infinity, label: "Явный дипфейк", cls: "deny" },
];

function riskBand(score) {
  const value = typeof score === "number" ? score : 0.5;
  return RISK_BANDS.find((band) => value < band.max) || RISK_BANDS[RISK_BANDS.length - 1];
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
