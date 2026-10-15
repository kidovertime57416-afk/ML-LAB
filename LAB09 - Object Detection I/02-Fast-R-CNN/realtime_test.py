from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "outputs"

from inference import webcam_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--regions", type=int, default=25)
    args = parser.parse_args()
    webcam_demo(OUT, args.camera, args.threshold, args.regions)


if __name__ == "__main__":
    main()
