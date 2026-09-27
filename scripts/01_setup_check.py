"""Step 01: check the Python packages and create the data folders (00 checks the raw files).

Run: python scripts/01_setup_check.py
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys

import sys as _sys
from pathlib import Path as _Path

# repo root on the path, for config and src
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("setup")

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "torch", "networkx", "tqdm"]
# only probed with find_spec, since importing kaggle tries to authenticate
OPTIONAL = {"kaggle": "00_download_data.py"}


def _check_packages() -> bool:
    ok = True
    for pkg in REQUIRED:
        try:
            m = importlib.import_module(pkg)
            log.info("  [ok] %-12s %s", pkg, getattr(m, "__version__", "?"))
        except ImportError:
            log.error("  [MISSING] %s  (pip install -r requirements.txt)", pkg)
            ok = False
    for pkg, used_by in OPTIONAL.items():
        found = importlib.util.find_spec(pkg) is not None
        log.info("  [opt] %-12s %s (only needed by %s)", pkg,
                 "installed" if found else "not installed", used_by)
    return ok


def main() -> int:
    log.info("Python %s", sys.version.split()[0])
    config.ensure_dirs()
    ok = _check_packages()
    log.info("Processed output dir: %s", config.DATA_PROCESSED)
    if not ok:
        log.error("Missing required packages; install them before proceeding.")
        return 1
    log.info("Environment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
