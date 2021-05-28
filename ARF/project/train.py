from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from dataset import SIIMDataset
from model import BigModel, Model2Branches, Model9Features
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

seed_everything(17)

CSV_DIR = Path("data")
train_df = pd.read_csv(CSV_DIR / "train_clean.csv")

IMAGE_DIR_TRAINING = Path("data")

# CSV_DIR = Path("/mnt/kingston/datasets/siim-isic-melanoma-classification")
CSV_DIR = Path("data")
# test_df = pd.read_csv(CSV_DIR / "test.csv")
test_df = pd.read_csv(CSV_DIR / "test_clean.csv")

# IMAGE_DIR = Path('/kaggle/input/siim-isic-melanoma-classification/jpeg')  # Use this when training with original images
# IMAGE_DIR_TEST = Path("/mnt/kingston/datasets/siim-isic-melanoma-classification/jpeg")


for u in range(3, 8):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="runs/" + "efficientnet-b" + str(u),
        filename="{epoch:02d}_{val_auc:.4f}",
        save_top_k=3,
        monitor="val_auc",
        mode="max",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auc", min_delta=0.00, patience=2, verbose=True, mode="max"
    )

    max_epochs = 20
    arch = f"efficientnet-b{u}"
    gpus = 1 if torch.cuda.is_available() else None
    # gpus = None
    tb_logger = pl_loggers.TensorBoardLogger(f"lightning_logs/", name=arch)

    trainer = pl.Trainer(
        gpus=gpus,
        precision=16 if gpus else 32,
        max_epochs=max_epochs,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback],
        logger=tb_logger,
        #limit_train_batches=10,
        #limit_val_batches=10,  # Debugging purposes
    )
    model = BigModel(train_df, test_df, IMAGE_DIR_TRAINING, IMAGE_DIR_TRAINING, arch)
    #model = Model2Branches(train_df, test_df, IMAGE_DIR_TRAINING, arch, n_meta_features=12)
    #model = Model9Features(
    #    train_df, test_df, IMAGE_DIR_TRAINING, arch, n_meta_features=12, image_size = 224
    #)

    trainer.fit(model)
