import shutil
import random
from pathlib import Path


SRC_DIR   = Path("/home/shared/imagecas/projections")
DST_DIR   = Path("/home/shared/imagecas/projections_split")
TRAIN_DIR = DST_DIR / "train"
VAL_DIR   = DST_DIR / "val"
VAL_RATIO = 0.2
RANDOM_SEED = 42


shutil.rmtree(DST_DIR, ignore_errors=True)
DST_DIR.mkdir(exist_ok=True)
TRAIN_DIR.mkdir(exist_ok=True)
VAL_DIR.mkdir(exist_ok=True)
random.seed(RANDOM_SEED)


def split_and_copy(prefix: str):
    files = sorted(SRC_DIR.glob(f"{prefix}*.joblib"))
    random.shuffle(files)
    val_count = int(len(files) * VAL_RATIO)
    val_files   = files[:val_count]
    train_files = files[val_count:]

    for f in train_files:
        shutil.copy(str(f), TRAIN_DIR / f.name)
    for f in val_files:
        shutil.copy(str(f), VAL_DIR / f.name)

    print(f"Copied {len(train_files)} files to {TRAIN_DIR}")
    print(f"Copied {len(val_files)} files to {VAL_DIR}")


split_and_copy("left")
split_and_copy("right")
