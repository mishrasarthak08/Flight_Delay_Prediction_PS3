"""Download and verify the exact public Kaggle snapshot used by the experiment."""

import argparse
import hashlib
import io
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA_PATH

SOURCE = "https://www.kaggle.com/api/v1/datasets/download/matinsajadi/flights"
EXPECTED_SHA256 = "8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa"


def verify(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"Dataset hash mismatch: {digest}. Do not silently replace the experiment data."
        )
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        print(f"Verified existing dataset: {verify(args.output)}")
        return
    if args.verify_only:
        raise FileNotFoundError(
            f"Missing dataset: {args.output}. Run this command without --verify-only."
        )
    request = urllib.request.Request(
        SOURCE, headers={"User-Agent": "flight-delay-capstone/2.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    content = archive.read("flights.csv")
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError(
            "Kaggle snapshot changed. The source does not match the pinned experiment hash."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(f"Downloaded and verified: {verify(args.output)}")


if __name__ == "__main__":
    main()
