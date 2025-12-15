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
    "Return up to three concise English words that describe the given list of activities.\n"
    "Do not add punctuation or explanations.\n"
    "Use the vocabulary that is common in the process mining domain."
)

load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
_client = OpenAI(api_key=_API_KEY) if _API_KEY else None

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
    """Heuristically label a subprocess using activity names."""
    clean_acts = [
        str(act).strip()
        for act in activities_list
        if act is not None and str(act).strip() not in ACTIVITY_IGNORE
    ]

    if not clean_acts:
        return "System Logic"

    unique_acts = list(dict.fromkeys(clean_acts))
    count = len(unique_acts)

    if count == 1:
        return unique_acts[0]

    if count <= 3:
        return " -> ".join(unique_acts)

    tokens_per_activity = [_tokenize(act) for act in unique_acts]
    flattened = [token for tokens in tokens_per_activity for token in tokens]

    if flattened:
        word_counter = Counter(flattened)
        for word, _freq in word_counter.most_common():
            coverage = sum(1 for tokens in tokens_per_activity if word in tokens)
            if coverage >= max(2, int(count * 0.4)):
                return f"{word.capitalize()} Phase"

    return f"{unique_acts[0]} -> {unique_acts[-1]}"


def name_subprocesses_with_gpt(activities: Sequence[str]) -> str:
    """Call gpt-5-nano to generate a short subprocess label."""
    if _client is None:
        return name_subprocesses(activities)

    unique = [act for act in dict.fromkeys(a.strip() for a in activities if a)]
    if not unique:
        return "System Logic"

    cache_key = " | ".join(unique)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    prompt = "Activities: " + "; ".join(unique[:20])
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
