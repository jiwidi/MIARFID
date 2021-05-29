import pandas as pd
import pytorch_lightning as pl
import torch
import os
import re
from pathlib import Path
from model import BigModel, BigModel9Features, Model9Features


def main():
    # FOlder to store predictions
    dir = os.path.dirname("runs/predictions/")
    if not os.path.exists(dir):
        os.makedirs(dir)
    # # test_df = pd.read_csv(CSV_DIR / "train_concat.csv")
    # CSV_DIR = Path("/mnt/kingston/datasets/siim-isic-melanoma-classification")
    # train_df = pd.read_csv(CSV_DIR / "train.csv")
    # test_df = pd.read_csv(CSV_DIR / "test.csv")
    # # IMAGE_DIR = Path('/kaggle/input/siim-isic-melanoma-classification/jpeg')  # Use this when training with original images
    # IMAGE_DIR_TEST = Path(
    #     "/mnt/kingston/datasets/siim-isic-melanoma-classification/jpeg"
    # )

    IMAGE_DIR_TRAINING = Path("data")
    CSV_DIR = Path("data")
    train_df = pd.read_csv(CSV_DIR / "train_full.csv")
    test_df = pd.read_csv(CSV_DIR / "test_full.csv")

    run_folder = "runs/"
    predictions = None
    for u in range(3,4):
        arch = f"efficientnet-b{u}"
        best_checkpoint = None
        best_checkpoint_auc = 0
        for checkpoint in os.listdir(run_folder + arch):
            auc = float(re.findall(r"(\d\.\d+(?=\.))", checkpoint)[0])
            if auc > best_checkpoint_auc:
                best_checkpoint = checkpoint
                best_checkpoint_auc = auc
        print(f"For arch {arch} loading checkpoint {best_checkpoint}")
        model = Model9Features(
            train_df, test_df, IMAGE_DIR_TRAINING, arch, n_meta_features=12, image_size=224
        )
        #model = BigModel9Features(train_df, test_df, IMAGE_DIR_TRAINING, IMAGE_DIR_TRAINING, arch, include_2019=True, image_size=224)
        model = model.load_from_checkpoint(
            run_folder + arch + "/" + best_checkpoint,
            train_df=train_df,
            test_df=test_df,
            image_dir=IMAGE_DIR_TRAINING,
            arch=arch,
            n_meta_features=12,
            image_size=224
        )

        gpus = 1 if torch.cuda.is_available() else None

        trainer = pl.Trainer(gpus=gpus)
        trainer.test(model)
        prediction = pd.read_csv(f"runs/predictions/submission_{arch}.csv")
        if not isinstance(predictions, pd.DataFrame):
            predictions = prediction
        else:
            predictions["target-" + arch] = prediction.target

    predictions.to_csv("runs/ensemble_prediction.csv", index=False)

    # Ensemble
    aux = []
    for column in predictions.columns:
        if "target" in column:
            aux.append(column)
    print(f"Ensembling columns {aux}")
    predictions["target"] = predictions[aux].sum(axis=1) / len(aux)
    predictions[["image_name", "target"]].round(6).to_csv(
        "runs/ensemble_prediction_mean.csv", index=False
    )


if __name__ == "__main__":
    main()
