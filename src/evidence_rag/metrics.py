from __future__ import annotations

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    """Hotpot/SQuAD-style English normalization; see official HotpotQA evaluator."""
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(answer: str | None, golds: tuple[str, ...]) -> tuple[float, float]:
    if answer is None:
        return 0.0, 0.0
    prediction = normalize(answer)
    scores = []
    for gold in golds:
        target = normalize(gold)
        exact = float(prediction == target)
        if prediction != target and ({prediction, target} & {"yes", "no", "noanswer"}):
            scores.append((exact, 0.0))
            continue
        common = sum((Counter(prediction.split()) & Counter(target.split())).values())
        f1 = 2 * common / (len(prediction.split()) + len(target.split())) if common else exact
        scores.append((exact, f1))
    return max(s[0] for s in scores), max(s[1] for s in scores)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
