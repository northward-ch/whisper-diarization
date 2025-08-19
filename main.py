#!/usr/bin/env python3
"""
main.py — Folder watcher that diarizes all audio files it finds.

- Periodically scans a watch folder.
- Waits until files are stable (size not changing).
- Runs: python diarize.py -a <audio_path>
- Moves successfully processed files into data/processed/rt_today/audio.
- Creates .processing marker while running to avoid duplicates.

Usage:
  python main.py --watch-dir /workspace/video_watch_service/raw/rt_today/audio \
                 --interval 10 \
                 --extensions .wav,.mp3 \
                 --dest-dir /workspace/data/processed/rt_today/audio
"""

import argparse
import concurrent.futures
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Set, Tuple

DEFAULT_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
STOP_EVENT = threading.Event()

def parse_extensions(ext_str: str) -> Tuple[str, ...]:
    if not ext_str:
        return DEFAULT_EXTS
    exts = []
    for raw in ext_str.split(","):
        s = raw.strip()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        exts.append(s.lower())
    return tuple(exts) if exts else DEFAULT_EXTS

def is_candidate(path: Path, exts: Tuple[str, ...]) -> bool:
    return path.is_file() and path.suffix.lower() in exts

def marker_paths(path: Path) -> Tuple[Path, Path]:
    proc = path.with_suffix(path.suffix + ".processing")
    done = path.with_suffix(path.suffix + ".done")
    return proc, done

def already_processed(path: Path) -> bool:
    proc_marker, done_marker = marker_paths(path)
    return done_marker.exists()

def diarize_one(python_bin: str, script_dir: Path, audio_path: Path) -> int:
    cmd = [python_bin, str(script_dir / "diarize.py"), "-a", str(audio_path)]
    logging.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.stdout:
            logging.debug("diarize.py output for %s:\n%s", audio_path.name, result.stdout)
        return result.returncode
    except Exception as e:
        logging.exception("Failed to run diarize.py: %s", e)
        return 1

def process_file(python_bin: str, script_dir: Path, audio_path: Path, dest_dir: Path) -> None:
    proc_marker, done_marker = marker_paths(audio_path)
    if done_marker.exists():
        logging.info("Skipping (already done): %s", audio_path.name)
        return
    try:
        proc_marker.touch(exist_ok=False)
    except FileExistsError:
        logging.info("Skipping (already processing): %s", audio_path.name)
        return

    try:
        rc = diarize_one(python_bin, script_dir, audio_path)
        if rc == 0:
            done_marker.touch(exist_ok=True)
            # Move original file into processed folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / audio_path.name
            audio_path.replace(target)
            logging.info("Completed diarization and moved %s → %s", audio_path.name, target)
        else:
            logging.error("Diarization failed: %s", audio_path.name)
    finally:
        proc_marker.unlink(missing_ok=True)

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
    parser = argparse.ArgumentParser(description="Periodic folder watcher for diarization.")
    parser.add_argument("--watch-dir", type=str, required=True,
                        help="Directory to watch for audio files.")
    parser.add_argument("--interval", type=int, default=10,
                        help="Polling interval in seconds.")
    parser.add_argument("--extensions", type=str, default=",".join(DEFAULT_EXTS),
                        help="Comma-separated list of audio extensions.")
    parser.add_argument("--dest-dir", type=str, required=True,
                        help="Destination folder to move processed files.")
    parser.add_argument("--python-binary", type=str, default=sys.executable,
                        help="Python interpreter to use.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    setup_signal_handlers()
    watch_dir = Path(args.watch_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    dest_dir = Path(args.dest_dir).resolve()
    exts = parse_extensions(args.extensions)

    logging.info("Watching %s → processed files go to %s", watch_dir, dest_dir)

    size_cache: Dict[Path, int] = {}
    stability_hits: Dict[Path, int] = {}
    STABILITY_REQUIRED = 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        futures: Set[concurrent.futures.Future] = set()
        while not STOP_EVENT.is_set():
            try:
                candidates = scan_candidates(watch_dir, exts)
                for path in candidates:
                    if already_processed(path):
                        continue
                    try:
                        current_size = path.stat().st_size
                    except FileNotFoundError:
                        continue
                    prev = size_cache.get(path)
                    if prev is not None and current_size == prev:
                        stability_hits[path] = stability_hits.get(path, 0) + 1
                    else:
                        stability_hits[path] = 0
                    size_cache[path] = current_size
                    if stability_hits[path] >= STABILITY_REQUIRED:
                        logging.info("Queued: %s", path.name)
                        futures.add(pool.submit(process_file, args.python_binary, script_dir, path, dest_dir))
                        stability_hits[path] = 0
                time.sleep(args.interval)
            except Exception:
                logging.exception("Watcher loop error.")
                time.sleep(args.interval)

    logging.info("Stopped.")

if __name__ == "__main__":
    main()
