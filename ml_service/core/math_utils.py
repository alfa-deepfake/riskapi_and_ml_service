from __future__ import annotations

import math
from statistics import fmean


def clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    n = min(len(left), len(right))
    if n < 3:
        return None
    a = left[:n]
    b = right[:n]
    mean_a = fmean(a)
    mean_b = fmean(b)
    da = [value - mean_a for value in a]
    db = [value - mean_b for value in b]
    denom_a = math.sqrt(sum(value * value for value in da))
    denom_b = math.sqrt(sum(value * value for value in db))
    denom = denom_a * denom_b
    if denom <= 1e-9:
        return None
    return sum(x * y for x, y in zip(da, db)) / denom


def best_lagged_correlation(left: list[float], right: list[float], max_lag: int = 1) -> tuple[float | None, int]:
    best_corr: float | None = None
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = left[-lag:]
            b = right[: len(a)]
        elif lag > 0:
            a = left[: len(left) - lag]
            b = right[lag:]
        else:
            a = left
            b = right
        corr = pearson_correlation(a, b)
        if corr is None:
            continue
        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_corr, best_lag


def _normalize_phrase(value: str) -> str:
    # Whisper punctuates freely ("Банк, сигнал, река.") and ё/е spelling varies
    # by voice — neither should count against the challenge-phrase match.
    value = value.lower().replace("ё", "е")
    value = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in value)
    return " ".join(value.split())


def phrase_word_match_fraction(expected: str, actual: str) -> float:
    """Fraction of expected words found in the transcript by per-word fuzzy
    match. Robust to ASR dropping/mangling one word of three on a poor mic; a
    transcript that recites extra vocabulary to sweep up matches overshoots
    the length guard and scores 0."""
    expected_words = _normalize_phrase(expected).split()
    actual_words = _normalize_phrase(actual).split()
    if not expected_words or not actual_words:
        return 0.0
    if len(actual_words) > 3 * len(expected_words):
        return 0.0
    matched = sum(
        1
        for word in expected_words
        if max(levenshtein_ratio(word, candidate) for candidate in actual_words) >= 0.70
    )
    return matched / len(expected_words)


def levenshtein_ratio(expected: str, actual: str) -> float:
    expected = _normalize_phrase(expected)
    actual = _normalize_phrase(actual)
    if expected == actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    previous = list(range(len(actual) + 1))
    for i, ca in enumerate(expected, start=1):
        current = [i]
        for j, cb in enumerate(actual, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    distance = previous[-1]
    return clamp01(1.0 - distance / max(len(expected), len(actual)))
