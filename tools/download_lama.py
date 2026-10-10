"""Download the big-lama TorchScript model into checkpoints/ (idempotent)."""
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "checkpoints" / "big-lama.pt"
URLS = [
    # IOPaint's official mirror (github release asset, no auth).
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
    # Official saicinpainting export, HF mirrors (may 401).
    "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
    "https://huggingface.co/SuperTanks007/big-lama/resolve/main/big-lama.pt",
]

if DEST.exists() and DEST.stat().st_size > 100_000_000:
    print("already present:", DEST.stat().st_size)
    sys.exit(0)

DEST.parent.mkdir(parents=True, exist_ok=True)
tmp = DEST.with_suffix(".part")
last_err = None
for url in URLS:
    try:
        print("trying", url, flush=True)
        urllib.request.urlretrieve(url, tmp)
        size = tmp.stat().st_size
        print("downloaded", size, flush=True)
        if size > 100_000_000:
            tmp.replace(DEST)
            print("saved ->", DEST, flush=True)
            sys.exit(0)
        last_err = f"too small: {size}"
    except Exception as exc:  # noqa: BLE001
        last_err = repr(exc)
        print("failed:", last_err, flush=True)
print("ALL URLS FAILED:", last_err, flush=True)
sys.exit(1)
