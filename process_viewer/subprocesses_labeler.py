"""Utilities for naming subprocesses, heuristically and with GPT."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Sequence

from dotenv import load_dotenv
from openai import OpenAI

ACTIVITY_IGNORE = {"+", "X", "O", "seq", "xor", "and", "loop", "None", "", "tau", "System Step"}
STOPWORDS = {"analyze", "check", "start", "end", "complete", "create", "process"}

_SYSTEM_PROMPT = (
    "You name subprocesses in process mining models.\n"
    "The activity list is unordered and may include repetitions or variants.\n"
    "Return 2-3 concise English words that capture the shared intent or phase.\n"
    "Use Title Case words like 'Repair Cycle' (capitalize each word).\n"
    "Do not add punctuation, numbering, or explanations.\n"
    "Prefer domain terms (e.g., registration, analysis, repair, notification).\n"
    "Avoid listing activity names verbatim.\n"
    "Never reuse the same label for different activity sets; if a label already exists, refine it to keep labels unique and informative."
)

load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
_client = OpenAI(api_key=_API_KEY) if _API_KEY else None
print("OPENAI_API_KEY loaded:", bool(_API_KEY))
print("OpenAI client created:", _client is not None)

_CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "subprocesses_labels.json"
if _CACHE_FILE.exists():
    try:
        _CACHE: Dict[str, str] = json.loads(_CACHE_FILE.read_text())
    except json.JSONDecodeError:
        _CACHE = {}
else:
    _CACHE = {}


def _tokenize(label: str) -> List[str]:
    """Split a label into meaningful lowercase tokens."""
    if not label:
        return []
    tokens = re.findall(r"[A-Za-z]+", label.lower())
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 3]


def _normalize_activity(act: str) -> str:
    """Trim and coerce activity to string, keeping empty string out."""
    return str(act).strip()


def _cache_key_with_counts(activities: Sequence[str]) -> str:
    """Stable cache key that differentiates inputs by multiplicity."""
    counts = Counter(_normalize_activity(a) for a in activities if _normalize_activity(a))
    return " | ".join(f"{act}:{counts[act]}" for act in sorted(counts))


def _existing_labels_hint(max_items: int = 10) -> str:
    """Compact hint with cached labels to guide uniqueness in the prompt."""
    if not _CACHE:
        return ""
    items = list(_CACHE.items())[:max_items]
    return "; ".join(f"{k} -> {v}" for k, v in items)


def _ensure_unique_label(label: str, cache_key: str) -> str:
    """Guarantee that a label is not reused for a different activity set."""
    duplicates = [k for k, v in _CACHE.items() if v.lower() == label.lower() and k != cache_key]
    if not duplicates:
        return label
    # Suffix avoids collision while keeping intent visible to the user.
    return f"{label} #{len(duplicates) + 1}"


def name_subprocesses(activities_list: Sequence[str]) -> str:
    """Heuristically label a subprocess using activity names (New Refactored Logic)."""
    clean_acts = [str(a) for a in activities_list if a and str(a).strip() not in ACTIVITY_IGNORE]
    if not clean_acts:
        return "Unknown Block"

    # Simple cases
    if len(clean_acts) == 1:
        return clean_acts[0]
    if len(clean_acts) <= 2:
        return " & ".join(clean_acts)

    # Heuristic for larger groups
    first, last = clean_acts[0], clean_acts[-1]

    # Find common theme words
    all_words = " ".join(clean_acts).lower().split()
    meaningful_words = [w for w in all_words if w not in STOPWORDS and len(w) > 3]

    theme = ""
    if meaningful_words:
        common_word, freq = Counter(meaningful_words).most_common(1)[0]
        # If the word appears often enough, use it as a title
        if freq >= 2:
            theme = f"[{common_word.capitalize()} Phase]"

    count_str = f"({len(clean_acts)} steps)"
    return f"{theme}\n{first} ... {last}\n{count_str}"


def name_subprocesses_with_gpt(activities: Sequence[str]) -> str:
    """Call gpt-5-nano to generate a short subprocess label."""
    if _client is None:
        return name_subprocesses(activities)

    cache_key = _cache_key_with_counts(activities)
    if not cache_key:
        return "System Logic"

    if cache_key in _CACHE:
        return _CACHE[cache_key]

    counts = Counter(_normalize_activity(a) for a in activities if _normalize_activity(a))
    prompt = "Activities (unordered, with counts): " + "; ".join(
        f"{act} x{counts[act]}" for act in sorted(counts)
    )

    labels_hint = _existing_labels_hint()
    if labels_hint:
        prompt += "\nExisting labels: " + labels_hint
    print(f"GPT Prompt: {prompt}")
    try:
        response = _client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
        )

        text = response.choices[0].message.content.strip()
        print(f"GPT Response: {text}")
        label = _ensure_unique_label(text[:60] or "System Logic", cache_key)

        try:
            if text:
                _CACHE[cache_key] = label
                _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                _CACHE_FILE.write_text(json.dumps(_CACHE, indent=4, ensure_ascii=False))
        except OSError:
            pass

        return label
    except Exception as e:
        print("GPT call failed, falling back to heuristic naming.")
        print(f"Error: {str(e)}")
        return name_subprocesses(activities)
