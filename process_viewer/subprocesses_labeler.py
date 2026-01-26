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
    "Avoid listing activity names verbatim."
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

    unique = sorted({a.strip() for a in activities if a})
    if not unique:
        return "System Logic"

    cache_key = " | ".join(unique)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    prompt = "Activities (unordered): " + "; ".join(unique[:20])
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
        label = text[:60] or "System Logic"

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
