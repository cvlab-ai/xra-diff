# prepare a pilot dataset for tests
import shutil
import random
from pathlib import Path

BASE_DIR   = Path("/home/shared/imagecas/projections_split")
VAL_DIR    = BASE_DIR / "val"
PILOT_DIR  = BASE_DIR / "pilot"
PILOT_RATIO   = 0.1
RANDOM_SEED = 42

shutil.rmtree(PILOT_DIR, ignore_errors=True)
PILOT_DIR.mkdir(exist_ok=True)


def make_pilot(prefix):
    files = sorted(VAL_DIR.glob(f"{prefix}*.joblib"))
    random.seed(RANDOM_SEED)
    random.shuffle(files)
    pilot_count = max(1, int(len(files) * PILOT_RATIO))
    pilot_files = files[:pilot_count]
    for f in pilot_files:
        shutil.copy(str(f), PILOT_DIR / f.name)

make_pilot("right")
make_pilot("left")
