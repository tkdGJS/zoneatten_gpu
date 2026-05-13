#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json"
TARGET = ROOT / "data" / "ShareGPT_V3_multiturn10_small_4096chars.json"
MAX_USER_TURNS = 10
MAX_FORMATTED_CHARS = 4096


def build_turn_records(conversations):
    records = []
    pending_user = None
    for message in conversations:
        role = message.get("from")
        value = (message.get("value") or "").strip()
        if not value:
            continue
        if role == "human":
            pending_user = value
        elif role == "gpt" and pending_user is not None:
            records.append((pending_user, value))
            pending_user = None
    return records


def formatted_length(records, limit):
    text = "<|begin_of_text|>"
    for user_text, assistant_text in records[:limit]:
        text += f"Human: {user_text}\n\n"
        text += f"Assistant: {assistant_text}\n\n"
    return len(text)


def main():
    with SOURCE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    subset = []
    for item in data:
        conversations = item.get("conversations") or []
        records = build_turn_records(conversations)
        if len(records) < MAX_USER_TURNS:
            continue
        if formatted_length(records, MAX_USER_TURNS) > MAX_FORMATTED_CHARS:
            continue
        subset.append(item)

    with TARGET.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)

    print(f"source={SOURCE}")
    print(f"target={TARGET}")
    print(f"selected_sessions={len(subset)}")


if __name__ == "__main__":
    main()
