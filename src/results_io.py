"""Locked read-merge-write for result JSONs shared by parallel runs of scripts 06 and 07.

Without the lock, two runs finishing together would drop each other's keys.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def merge_json(path: Path, new: Dict, timeout_s: float = 600.0,
               stale_s: float = 3600.0) -> Tuple[Dict, List[str]]:
    """Merge `new` into the JSON object at `path` under a lock.

    Returns (merged, replaced), where replaced lists the keys that were overwritten.
    """
    path = Path(path)
    lock = path.with_name(path.name + ".lock")
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_s:   # left by a killed process
                    logger.warning("removing stale lock %s", lock)
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"could not lock {lock} within {timeout_s:.0f}s")
            time.sleep(0.2)
    try:
        merged = json.loads(path.read_text()) if path.exists() else {}
        replaced = sorted(set(merged) & set(new))
        merged.update(new)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(merged, indent=2))
        os.replace(tmp, path)                                   # atomic on the same volume
        return merged, replaced
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
