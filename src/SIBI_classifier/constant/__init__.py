from pathlib import Path
from glob import glob
import os

ROOT_DIR = Path(__file__).resolve().parents[3]

CONFIG_FILE_PATH = Path(glob(os.path.join(ROOT_DIR, "**", "config.yaml"), recursive=True)[0])
PARAMS_FILE_PATH = Path(glob(os.path.join(ROOT_DIR, "**", "params.yaml"), recursive=True)[0])
