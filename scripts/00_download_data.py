"""Step 00: put every raw dataset in data/raw/, downloading those that have a public link.

IEEE-CIS, Elliptic and Ethereum come through the Kaggle API (pip install kaggle, credentials
set up as https://www.kaggle.com/docs/api describes, and for IEEE-CIS the competition rules
accepted on its page); Amazon, YelpChi and S-FFSD are direct downloads. Elliptic++ (Google
Drive) and DGraph-Fin (sign-up) are placed by hand; for them the script prints the steps.
A dataset whose files are already in place is skipped, and the script exits with an error
while any is missing.

Run: python scripts/00_download_data.py [--only ieee_cis gad elliptic ellipticpp dgraph s_ffsd ethereum]
"""
from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import sys as _sys
from pathlib import Path as _Path

# repo root on the path, for config and src
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("download")

# Kaggle sources: (kind, slug)
KAGGLE = {
    "ieee_cis": ("competition", "ieee-fraud-detection"),
    "elliptic": ("dataset", "ellipticco/elliptic-data-set"),
    "ethereum": ("dataset", "xblock/ethereum-phishing-transaction-network"),
}
# direct downloads: zip archives holding the files of config.RAW_FILES
URLS = {
    "gad": ["https://raw.githubusercontent.com/YingtongDou/CARE-GNN/refs/heads/master/data/Amazon.zip",
            "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/refs/heads/master/data/YelpChi.zip"],
    "s_ffsd": ["https://raw.githubusercontent.com/AI4Risk/antifraud/refs/heads/main/data/S-FFSD.zip"],
}
MANUAL = {
    "ieee_cis": "accept the rules at https://www.kaggle.com/c/ieee-fraud-detection/rules, then "
                "download the data on that page",
    "elliptic": "download https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
    "ethereum": "download https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network",
    "gad": "download Amazon.zip and YelpChi.zip from https://github.com/YingtongDou/CARE-GNN "
           "(folder data/) and unzip them",
    "s_ffsd": "download data/S-FFSD.zip from https://github.com/AI4Risk/antifraud and unzip it",
    "ellipticpp": "open the Google Drive folder linked from https://github.com/git-disl/EllipticPlusPlus "
                  "and download the three files of 'Transactions Dataset'",
    "dgraph": "sign up at https://dgraph.xinye.com, download DGraphFin.zip from its Datasets page "
              "(DGraph-Fin) and unzip it",
}


def _missing(name: str) -> list:
    raw_dir, files = config.RAW_FILES[name]
    return [f for f in files if not (raw_dir / f).exists()]


def _fetch_kaggle(name: str, out: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi   # importing kaggle reads the API token

    api = KaggleApi()
    api.authenticate()
    kind, slug = KAGGLE[name]
    if kind == "competition":
        api.competition_download_files(slug, path=str(out))
    else:
        api.dataset_download_files(slug, path=str(out))


def _fetch_urls(name: str, out: Path) -> None:
    for url in URLS[name]:
        log.info("[%s] %s", name, url)
        urllib.request.urlretrieve(url, out / url.rsplit("/", 1)[-1])


def _extract(archives: Iterable[Path], wanted: Iterable[str], raw_dir: Path) -> None:
    """Copy the wanted files, found by name at any depth of the archives, into raw_dir."""
    wanted = set(wanted)
    for z in archives:
        with zipfile.ZipFile(z) as zf:
            for member in zf.infolist():
                base = Path(member.filename).name
                if base in wanted and not member.is_dir():
                    with zf.open(member) as src, open(raw_dir / base, "wb") as dst:
                        shutil.copyfileobj(src, dst)


def fetch(name: str) -> bool:
    """Make sure the raw files of one source are in place; True if they are."""
    raw_dir, files = config.RAW_FILES[name]
    if not _missing(name):
        log.info("[%s] present in %s", name, raw_dir)
        return True
    if name in KAGGLE or name in URLS:
        tmp = raw_dir / "_download"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            (_fetch_kaggle if name in KAGGLE else _fetch_urls)(name, tmp)
            _extract(sorted(tmp.glob("*.zip")), files, raw_dir)
        except Exception as e:  # noqa: BLE001  (no token, rules not accepted, network)
            log.error("[%s] download failed: %s", name, e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    missing = _missing(name)
    if missing:
        log.error("[%s] missing %s in %s: %s", name, ", ".join(missing), raw_dir, MANUAL[name])
        return False
    log.info("[%s] ready in %s", name, raw_dir)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the raw datasets or print how to get them")
    ap.add_argument("--only", nargs="+", choices=list(config.RAW_FILES), default=list(config.RAW_FILES))
    args = ap.parse_args()
    config.ensure_dirs()
    ok = [fetch(name) for name in args.only]
    if not all(ok):
        log.error("place the missing files as described above (README.md, Data) and rerun")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
