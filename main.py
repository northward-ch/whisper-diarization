#!/usr/bin/env python3
"""
main.py — Folder watcher using DiarizationEngine (models loaded once).

Behavior:
- Scans --watch-dir periodically for audio files.
- Waits for size stability to ensure the file is closed.
- On SUCCESS:
    * Writes <dest>/<stem>.txt and <dest>/<stem>.srt
    * Copies the audio as <dest>/<stem>.wav
    * Removes the original from --watch-dir
- On FAILURE:
    * Moves original into <dest>/errors/error_<name>.wav
    * Removes any partial markers in the source folder
- The input folder should be empty after each file is handled.

Usage example:
  python main.py \
    --watch-dir /workspace/video_watch_service/data/raw/rt_today/audio \
    --dest-dir  /workspace/whisper_diarization/data/processed/rt_today/audio \
    --interval  10 \
    --extensions .wav,.mp3 \
    --device cuda \
    --whisper-model medium.en \
    --batch-size 8 \
    --no-stem false
"""

import argparse
import concurrent.futures
import logging
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Set, Tuple

from diarize_engine import DiarizationEngine

DEFAULT_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
STOP_EVENT = threading.Event()

def parse_extensions(ext_str: str) -> Tuple[str, ...]:
    if not ext_str:
        return DEFAULT_EXTS
    items = []
    for raw in ext_str.split(","):
        s = raw.strip()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        items.append(s.lower())
    return tuple(items) if items else DEFAULT_EXTS

def should_skip_name(name: str) -> bool:
    return name.startswith("error_") or name.endswith(".processing") or name.endswith(".done")

def is_candidate(p: Path, exts: Tuple[str, ...]) -> bool:
    return p.is_file() and p.suffix.lower() in exts and not should_skip_name(p.name)

def marker_paths(p: Path):
    return p.with_suffix(p.suffix + ".processing"), p.with_suffix(p.suffix + ".done")

def scan_candidates(watch_dir: Path, exts: Tuple[str, ...]) -> Set[Path]:
    return {p for p in watch_dir.rglob("*") if is_candidate(p, exts)}

def setup_signal_handlers():
    def _handler(signum, _frame):
        logging.info("Received signal %s — shutting down…", signum)
        STOP_EVENT.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser(description="Folder watcher for diarization (persistent models).")
    ap.add_argument("--watch-dir", required=True, help="Directory to watch for audio files.")
    ap.add_argument("--dest-dir", required=True, help="Directory to write all outputs to.")
    ap.add_argument("--interval", type=int, default=10, help="Polling interval in seconds.")
    ap.add_argument("--extensions", type=str, default=",".join(DEFAULT_EXTS),
                    help="Comma-separated audio extensions (e.g., .wav,.mp3).")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="Max concurrent diarizations. Use 1 (recommended) due to model memory.")
    # Engine options
    ap.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu",
                    help="cuda or cpu")
    ap.add_argument("--whisper-model", type=str, default="medium.en",
                    help="Whisper model name")
    ap.add_argument("--batch-size", type=int, default=8, help="Batched inference size")
    ap.add_argument("--suppress-numerals", action="store_true", default=False,
                    help="Suppress numerical digits in transcription")
    ap.add_argument("--no-stem", action="store_true", default=False,
                    help="Disable source separation (Demucs)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    setup_signal_handlers()

    watch_dir = Path(args.watch_dir).resolve()
    dest_dir = Path(args.dest_dir).resolve()
    errors_dir = dest_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)

    exts = parse_extensions(args.extensions)

    if not watch_dir.exists():
        logging.info("Creating watch dir: %s", watch_dir)
        watch_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Watching: %s", watch_dir)
    logging.info("Outputs to: %s", dest_dir)
    logging.info("Extensions: %s", ", ".join(exts))
    logging.info("Device: %s | Whisper: %s | Batch: %d | Stemming: %s",
                 args.device, args.whisper_model, args.batch_size, "off" if args.no_stem else "on")

    # Load engine ONCE
    engine = DiarizationEngine(
        device=args.device,
        whisper_model_name=args.whisper_model,
        batch_size=args.batch_size,
        suppress_numerals=args.suppress_numerals,
        enable_stemming=not args.no_stem,
    )

    size_cache: Dict[Path, int] = {}
    stable_hits: Dict[Path, int] = {}
    STABLE_N = 2

    def handle_success(src_path: Path):
        # Remove original from watch-dir to keep it empty
        try:
            src_path.unlink(missing_ok=True)
        except Exception:
            logging.exception("Could not delete original after success: %s", src_path)

    def handle_failure(src_path: Path):
        # Move to dest/errors/error_<name>.wav
        err_target = errors_dir / f"error_{src_path.name}"
        try:
            if err_target.exists():
                err_target.unlink()
            shutil.move(str(src_path), str(err_target))
            logging.info("Moved error file to: %s", err_target)
        except Exception:
            logging.exception("Could not move failed file to errors: %s", src_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures: Set[concurrent.futures.Future] = set()

        while not STOP_EVENT.is_set():
            try:
                # Clean dropped files from caches
                for p in list(size_cache.keys()):
                    if not p.exists():
                        size_cache.pop(p, None)
                        stable_hits.pop(p, None)

                # Scan for new candidates
                for path in sorted(scan_candidates(watch_dir, exts)):
                    proc, done = marker_paths(path)
                    if done.exists() or proc.exists():
                        continue

                    # Size stability check
                    try:
                        cur = path.stat().st_size
                    except FileNotFoundError:
                        continue

                    prev = size_cache.get(path)
                    if prev is not None and cur == prev:
                        stable_hits[path] = stable_hits.get(path, 0) + 1
                    else:
                        stable_hits[path] = 0
                    size_cache[path] = cur

                    if stable_hits[path] >= STABLE_N:
                        # Mark processing
                        try:
                            proc.touch(exist_ok=False)
                        except FileExistsError:
                            continue

                        def _job(p=path, marker=proc):
                            try:
                                engine.process_one(p, dest_dir)
                            except Exception:
                                logging.exception("Diarization failed: %s", p.name)
                                handle_failure(p)
                            else:
