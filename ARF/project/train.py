from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from dataset import SIIMDataset
from model import BigModel
from pytorch_lightning import loggers as pl_loggers


CSV_DIR = Path("/Users/jaimeferrando/Downloads/siim-isic-melanoma-classification")
train_df = pd.read_csv(CSV_DIR / "train.csv")
test_df = pd.read_csv(CSV_DIR / "test.csv")
# IMAGE_DIR = Path('/kaggle/input/siim-isic-melanoma-classification/jpeg')  # Use this when training with original images
IMAGE_DIR = Path(
    "/Users/jaimeferrando/Downloads/siim-isic-melanoma-classification/jpeg"
)

for u in range(3, 8):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        "{epoch:02d}_{val_auc:.4f}", save_top_k=1, monitor="val_auc", mode="max"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auc", min_delta=0.00, patience=5, verbose=False, mode="max"
    )

    max_epochs = 50
    arch = f"efficientnet-b{u}"
    gpus = 1 if torch.cuda.is_available() else None
    tb_logger = pl_loggers.TensorBoardLogger(f"lightning_logs/", name=arch)

    trainer = pl.Trainer(
        gpus=gpus,
        precision=16 if gpus else 32,
        max_epochs=max_epochs,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=tb_logger,
    )
    model = BigModel(train_df, test_df, IMAGE_DIR, arch)

    trainer.fit(model)
