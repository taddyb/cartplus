"""Download the released CAMELS-US, CAMELS-AUS and CAMELS-GB input files into data/ and check their SHA256 checksums.

    python scripts/data/fetch_data.py [--owner OWNER] [--repo REPO] [--tag TAG] [--dest DIR]

Each file is fetched from https://github.com/<owner>/<repo>/releases/download/<tag>/<file>. Files already present with the right
checksum are skipped; a download whose checksum does not match is deleted and reported. Standard library only.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

OWNER = "xiaxilin"
REPO = "cartplus"
TAG = "v1.0"

SHA256 = {
    "camels_us_daymet671_v4full.npz": "84482d304b59aad8717e97eb7a90784bb08a25256ec45b1d6dac4de223139a54",
    "elev_bands671.npz": "f980712c6a64107cb63e50eced3b3d520fe0805890fdf66cadb12a46c532e0ae",
    "camels_aus_v2.npz": "2102e06ec706e93303b13acec613421b131a00532be99524f08551a4c2dfb928",
    "elev_bands_aus.npz": "afe36037b6d993c8b5bf8f064cdec72cef2c28c1dc27762bb3e114e5f616871f",
    "camels_gb_cart.npz": "454187814591e3e57f9d10d79b7fabe997f9b7aa30fab4bb958502b2c5206238",
}

DATA = Path(__file__).resolve().parents[2] / "data"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest):
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {dest.name}: {done / 2 ** 20:7.1f} / {total / 2 ** 20:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", default=OWNER)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--dest", default=str(DATA), help="target folder (default <repository>/data)")
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/{a.owner}/{a.repo}/releases/download/{a.tag}"
    bad = []
    for name, want in SHA256.items():
        path = dest / name
        if path.exists() and sha256(path) == want:
            print(f"{name}: present, checksum ok")
            continue
        print(f"{name}: downloading {base}/{name}")
        try:
            fetch(f"{base}/{name}", path)
        except Exception as e:                                                                 # network or HTTP error
            print(f"{name}: download failed ({e})")
            bad.append(name)
            continue
        got = sha256(path)
        if got != want:
            print(f"{name}: checksum mismatch (expected {want}, got {got}); file removed")
            path.unlink()
            bad.append(name)
        else:
            print(f"{name}: checksum ok")
    if bad:
        sys.exit(f"failed: {', '.join(bad)}")
    print(f"all files in {dest}")


if __name__ == "__main__":
    main()
