#!/usr/bin/env python3
import argparse, concurrent.futures, logging, os, signal, subprocess, sys, threading, time, shutil
from pathlib import Path
from typing import Dict, Set, Tuple

DEFAULT_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
STOP_EVENT = threading.Event()

def parse_extensions(ext_str: str) -> Tuple[str, ...]:
    if not ext_str: return DEFAULT_EXTS
    out=[]
    for raw in ext_str.split(","):
        s = raw.strip()
        if not s: continue
        if not s.startswith("."): s = "." + s
        out.append(s.lower())
    return tuple(out) if out else DEFAULT_EXTS

def should_skip_name(name: str) -> bool:
    # Skip error copies and hidden/meta files
    return name.startswith("error_") or name.endswith(".processing") or name.endswith(".done")

def is_candidate(p: Path, exts: Tuple[str, ...]) -> bool:
    return p.is_file() and (p.suffix.lower() in exts) and not should_skip_name(p.name)

def marker_paths(p: Path):
    return p.with_suffix(p.suffix + ".processing"), p.with_suffix(p.suffix + ".done")

def unique_path(base: Path) -> Path:
    """Return a non-existing path by appending -N before the suffix if needed."""
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    parent = base.parent
    i = 1
    while True:
        cand = parent / f"{stem}-{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def diarize_one(python_bin: str, script_dir: Path, audio_path: Path) -> int:
    cmd = [python_bin, str(script_dir / "diarize.py"), "-a", str(audio_path)]
    logging.info("Running: %s", " ".join(cmd))
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.stdout:
            logging.debug("diarize.py output for %s:\n%s", audio_path.name, r.stdout)
        return r.returncode
    except Exception as e:
        logging.exception("Failed to run diarize.py on %s: %s", audio_path, e)
        return 1

def process_file(python_bin: str, script_dir: Path, audio_path: Path, dest_dir: Path):
    proc, done = marker_paths(audio_path)
    if done.exists(): 
        logging.info("Skipping (already done): %s", audio_path.name); return
    try:
        proc.touch(exist_ok=False)
    except FileExistsError:
        logging.info("Skipping (already processing): %s", audio_path.name); return

    try:
        rc = diarize_one(python_bin, script_dir, audio_path)
        if rc == 0:
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / audio_path.name
            # If same-device, replace = move; handles overwrite safely after unlink
            if target.exists(): target.unlink()
            audio_path.replace(target)
            done.touch(exist_ok=True)
            logging.info("Completed diarization and moved %s -> %s", target.name, target)
        else:
            logging.error("Diarization failed: %s", audio_path.name)
            # Create error copy in the source folder, then delete original
            err_copy = audio_path.with_name(f"error_{audio_path.name}")
            err_copy = unique_path(err_copy)  # avoid clobbering prior errors
            try:
                shutil.copy2(audio_path, err_copy)
                logging.info("Wrote error copy: %s", err_copy.name)
            except Exception:
                logging.exception("Could not write error copy for %s", audio_path.name)
            # Remove original to avoid re-queuing
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                logging.exception("Could not delete original after error: %s", audio_path)
    finally:
        proc.unlink(missing_ok=True)

def scan_candidates(watch_dir: Path, exts: Tuple[str, ...]) -> Set[Path]:
    return {p for p in watch_dir.rglob("*") if is_candidate(p, exts)}

def setup_signal_handlers():
    def _h(signum, _):
        logging.info("Signal %s received, shutting down…", signum)
        STOP_EVENT.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(sig, _h)
        except Exception: pass

def main():
    ap = argparse.ArgumentParser(description="Periodic folder watcher for diarization.")
    ap.add_argument("--watch-dir", required=True, help="Folder with incoming audio.")
    ap.add_argument("--dest-dir",   required=True, help="Folder to move processed audio.")
    ap.add_argument("--interval", type=int, default=10, help="Polling interval (s).")
    ap.add_argument("--extensions", default=",".join(DEFAULT_EXTS),
                    help="Comma-separated audio extensions (e.g. .wav,.mp3).")
    ap.add_argument("--max-workers", type=int, default=1, help="Max concurrent jobs.")
    ap.add_argument("--python-binary", default=sys.executable,
                    help="Python interpreter to run diarize.py.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    setup_signal_handlers()
    watch = Path(args.watch_dir).resolve()
    dest  = Path(args.dest_dir).resolve()
    exts  = parse_extensions(args.extensions)
    script_dir = Path(__file__).resolve().parent

    if not watch.exists():
        logging.info("Creating watch dir: %s", watch); watch.mkdir(parents=True, exist_ok=True)

    logging.info("Watching: %s", watch)
    logging.info("Processed out: %s", dest)
    logging.info("Extensions: %s", ", ".join(exts))
    logging.info("Interval: %ss | Max workers: %s", args.interval, args.max_workers)

    size_cache: Dict[Path, int] = {}
    stable_hits: Dict[Path, int] = {}
    STABLE_N = 2  # require N consecutive equal sizes

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        inflight: Set[concurrent.futures.Future] = set()
        while not STOP_EVENT.is_set():
            try:
                # drop cache entries for removed files
                for p in list(size_cache.keys()):
                    if not p.exists():
                        size_cache.pop(p, None); stable_hits.pop(p, None)

                for path in sorted(scan_candidates(watch, exts)):
                    proc, done = marker_paths(path)
                    if done.exists() or proc.exists():
                        continue
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
                        logging.info("Queued: %s", path.name)
                        inflight.add(pool.submit(process_file, args.python_binary, script_dir, path, dest))
                        stable_hits[path] = 0

                done_futs, inflight = concurrent.futures.wait(
                    inflight, timeout=0, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for _ in done_futs: pass
                time.sleep(args.interval)
            except Exception:
                logging.exception("Watcher loop error; continuing.")
                time.sleep(args.interval)
    logging.info("Stopped.")

if __name__ == "__main__":
    main()
