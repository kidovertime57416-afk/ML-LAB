from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments import load_model as load_selected_model
from inference import detect_demo, draw


def load_model():
    # โน้ต: helper เดิมใช้โมเดล Aquarium ที่เลือกจาก validation แล้ว
    return load_selected_model(Path(__file__).resolve().parent / "outputs")
