from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "outputs"

from experiments import evaluate_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    evaluate_selected(OUT, args.split, args.limit, args.batch)


if __name__ == "__main__":
    main()
