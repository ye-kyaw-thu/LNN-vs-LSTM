import argparse
import os
import subprocess
import re

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/unipen/"
INDEX_FILE = "unipen-index.txt"

DATA_ROOT = "data"

PATTERNS = {
    "digit": re.compile(r"^[0-9].*\.dat\.gz$"),
    "lower": re.compile(r"^[a-z].*\.dat\.gz$"),
    "upper": re.compile(r"^[A-Z].*\.dat\.gz$")
}


def download_index():
    if not os.path.exists(INDEX_FILE):
        subprocess.run(
            ["wget", "-O", INDEX_FILE, BASE_URL],
            check=False
        )


def download_files(subset):
    out_dir = os.path.join(DATA_ROOT, subset)
    os.makedirs(out_dir, exist_ok=True)

    with open(INDEX_FILE) as f:
        for line in f:
            line = line.strip()
            if PATTERNS[subset].match(line):
                url = BASE_URL + line
                target = os.path.join(out_dir, line)

                if os.path.exists(target):
                    continue

                print(f"[DOWNLOADING] {line}")
                subprocess.run(
                    ["wget", "-c", url, "-O", target],
                    check=False
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["digit", "lower", "upper", "all"],
        required=True
    )
    args = parser.parse_args()

    download_index()

    subsets = ["digit", "lower", "upper"] if args.subset == "all" else [args.subset]

    for s in subsets:
        print(f"\n=== Downloading {s} ===")
        download_files(s)


if __name__ == "__main__":
    main()

