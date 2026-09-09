#!/usr/bin/env python3
"""Build a student-level Yousician ML dataset from raw exercise JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


INITIAL_START = 0.0
INITIAL_END = 7.0
POST_START = 15.0
POST_END = 30.0


@dataclass
class InitialStats:
    active_days: set[int] = field(default_factory=set)
    sessions: set[int] = field(default_factory=set)
    time_playing: float = 0.0
    total_exercises: int = 0
    notes_successful: float = 0.0
    notes_evaluated: float = 0.0
    chords_successful: float = 0.0
    chords_evaluated: float = 0.0
    difficulty_sum: float = 0.0
    difficulty_count: int = 0
    completed_count: int = 0
    abandoned_count: int = 0
    session_index_sum: float = 0.0
    session_index_count: int = 0
    songs: set[str] = field(default_factory=set)
    exercises: set[str] = field(default_factory=set)
    play_modes: Counter[str] = field(default_factory=Counter)


@dataclass
class AccuracyStats:
    successful: float = 0.0
    evaluated: float = 0.0


def stream_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[dict[str, Any]]:
    """Yield objects from a large top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False

    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_size)
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
                    item, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break

                if isinstance(item, dict):
                    yield item
                buffer = buffer[idx:]


def number(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def is_completed(record: dict[str, Any]) -> bool:
    return text(record.get("is_played_in_full")).lower() in {"t", "true", "1", "yes"}


def is_abandoned(record: dict[str, Any]) -> bool:
    return text(record.get("exit_status")).lower() in {"quit", "abandoned", "cancelled", "canceled"}


def update_initial(stats: InitialStats, record: dict[str, Any], day: float) -> None:
    stats.active_days.add(int(day))

    session_index = record.get("session_index")
    if session_index not in (None, ""):
        session = int(number(session_index))
        stats.sessions.add(session)
        stats.session_index_sum += session
        stats.session_index_count += 1

    stats.time_playing += number(record.get("time_playing"))
    stats.total_exercises += 1

    stats.notes_successful += number(record.get("notes_successful"))
    stats.notes_evaluated += number(record.get("notes_evaluated"))
    stats.chords_successful += number(record.get("chords_successful"))
    stats.chords_evaluated += number(record.get("chords_evaluated"))

    difficulty = record.get("difficulty_level")
    if difficulty not in (None, ""):
        stats.difficulty_sum += number(difficulty)
        stats.difficulty_count += 1

    if is_completed(record):
        stats.completed_count += 1
    if is_abandoned(record):
        stats.abandoned_count += 1

    song_id = text(record.get("song_id"))
    exercise_id = text(record.get("exercise_id"))
    play_mode = text(record.get("play_mode"))
    if song_id:
        stats.songs.add(song_id)
    if exercise_id:
        stats.exercises.add(exercise_id)
    if play_mode:
        stats.play_modes[play_mode] += 1


def update_accuracy(stats: AccuracyStats, record: dict[str, Any]) -> None:
    stats.successful += number(record.get("notes_successful")) + number(record.get("chords_successful"))
    stats.evaluated += number(record.get("notes_evaluated")) + number(record.get("chords_evaluated"))


def build_dataset(input_path: Path, output_path: Path) -> tuple[int, int]:
    initial_by_user: dict[str, InitialStats] = defaultdict(InitialStats)
    initial_accuracy: dict[str, AccuracyStats] = defaultdict(AccuracyStats)
    post_accuracy: dict[str, AccuracyStats] = defaultdict(AccuracyStats)
    records_read = 0

    for record in stream_json_array(input_path):
        records_read += 1
        user_id = text(record.get("user_id"))
        if not user_id:
            continue

        day = number(record.get("days_since_signup"), default=-1.0)
        if INITIAL_START <= day < INITIAL_END:
            update_initial(initial_by_user[user_id], record, day)
            update_accuracy(initial_accuracy[user_id], record)
        elif POST_START <= day <= POST_END:
            update_accuracy(post_accuracy[user_id], record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user_id",
        "total_days_active_initial",
        "number_of_sessions_initial",
        "total_time_playing_initial",
        "average_time_per_session_initial",
        "total_exercises_initial",
        "average_note_accuracy_initial",
        "average_chord_accuracy_initial",
        "average_difficulty_initial",
        "completion_rate_initial",
        "abandonment_rate_initial",
        "average_session_index_initial",
        "songs_practiced_initial",
        "exercises_practiced_initial",
        "play_mode_frequency_initial",
        "Performance_Improvement",
    ]

    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for user_id in sorted(initial_by_user):
            initial_perf = ratio(initial_accuracy[user_id].successful, initial_accuracy[user_id].evaluated)
            post_perf = ratio(post_accuracy[user_id].successful, post_accuracy[user_id].evaluated)
            if initial_perf is None or post_perf is None:
                continue

            stats = initial_by_user[user_id]
            sessions = len(stats.sessions)
            total = stats.total_exercises
            note_accuracy = ratio(stats.notes_successful, stats.notes_evaluated)
            chord_accuracy = ratio(stats.chords_successful, stats.chords_evaluated)
            play_mode_frequency = max(stats.play_modes.values()) / total if total else None

            writer.writerow(
                {
                    "user_id": user_id,
                    "total_days_active_initial": len(stats.active_days),
                    "number_of_sessions_initial": sessions,
                    "total_time_playing_initial": round(stats.time_playing, 6),
                    "average_time_per_session_initial": round(stats.time_playing / sessions, 6) if sessions else "",
                    "total_exercises_initial": total,
                    "average_note_accuracy_initial": round(note_accuracy, 6) if note_accuracy is not None else "",
                    "average_chord_accuracy_initial": round(chord_accuracy, 6) if chord_accuracy is not None else "",
                    "average_difficulty_initial": round(stats.difficulty_sum / stats.difficulty_count, 6)
                    if stats.difficulty_count
                    else "",
                    "completion_rate_initial": round(stats.completed_count / total, 6) if total else "",
                    "abandonment_rate_initial": round(stats.abandoned_count / total, 6) if total else "",
                    "average_session_index_initial": round(
                        stats.session_index_sum / stats.session_index_count, 6
                    )
                    if stats.session_index_count
                    else "",
                    "songs_practiced_initial": len(stats.songs),
                    "exercises_practiced_initial": len(stats.exercises),
                    "play_mode_frequency_initial": round(play_mode_frequency, 6)
                    if play_mode_frequency is not None
                    else "",
                    "Performance_Improvement": 1 if post_perf > initial_perf else 0,
                }
            )
            rows_written += 1

    return records_read, rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/yousician_ukulele.json"),
        help="Raw Yousician JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/yousician_processed.csv"),
        help="Processed student-level CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_read, rows_written = build_dataset(args.input, args.output)
    print(f"Read {records_read:,} raw records")
    print(f"Wrote {rows_written:,} student rows to {args.output}")


if __name__ == "__main__":
    main()
