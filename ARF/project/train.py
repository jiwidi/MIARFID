from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from dataset import SIIMDataset
from model import BigModel

CSV_DIR = Path("/mnt/kingston/datasets/siim-isic-melanoma-classification")
train_df = pd.read_csv(CSV_DIR / "train.csv")
test_df = pd.read_csv(CSV_DIR / "test.csv")
# IMAGE_DIR = Path('/kaggle/input/siim-isic-melanoma-classification/jpeg')  # Use this when training with original images
IMAGE_DIR = Path("/mnt/kingston/datasets/siim-isic-melanoma-classification/jpeg")


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    "{epoch:02d}_{val_auc:.4f}", save_top_k=1, monitor="val_auc", mode="max"
)

max_epochs = 50
gpus = 1 if torch.cuda.is_available() else None
trainer = pl.Trainer(
    gpus=gpus,
    precision=16 if gpus else 32,
    max_epochs=max_epochs,
    checkpoint_callback=checkpoint_callback,
)
model = BigModel(train_df, test_df, IMAGE_DIR)

trainer.fit(model)