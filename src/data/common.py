"""Shared data-pipeline definitions."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


INITIAL_START = 0.0
INITIAL_END = 7.0
FUTURE_START = 7.0
FUTURE_END = 31.0

RAW_COLUMNS = (
    "user_id",
    "song_id",
    "exercise_id",
    "days_since_signup",
    "song_type",
    "difficulty_level",
    "session_index",
    "play_mode",
    "chords_evaluated",
    "chords_successful",
    "notes_evaluated",
    "notes_successful",
    "time_playing",
    "is_played_in_full",
    "exit_status",
    "exercise_part_index",
)


def stream_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Yield dictionaries from a large top-level JSON array."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError("Expected a top-level JSON array")
                    buffer = buffer[1:]
                    started = True
                    continue

                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return

                try:
                    item, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break

                if isinstance(item, dict):
                    yield item
                buffer = buffer[index:]

    if started:
        raise ValueError("JSON array ended before a closing bracket was found")
