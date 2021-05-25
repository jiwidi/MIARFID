import pandas as pd
import pytorch_lightning as pl
import torch
import os
import re
from pathlib import Path
from model import BigModel, Model2Branches


def main():
    # FOlder to store predictions
    dir = os.path.dirname("runs/predictions/")
    if not os.path.exists(dir):
        os.makedirs(dir)
    # test_df = pd.read_csv(CSV_DIR / "train_concat.csv")
    CSV_DIR = Path("/mnt/kingston/datasets/siim-isic-melanoma-classification")
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    # IMAGE_DIR = Path('/kaggle/input/siim-isic-melanoma-classification/jpeg')  # Use this when training with original images
    IMAGE_DIR_TEST = Path(
        "/mnt/kingston/datasets/siim-isic-melanoma-classification/jpeg"
    )
    IMAGE_DIR_TRAINING = IMAGE_DIR_TEST
    run_folder = "runs/"
    predictions = None
    for u in range(3, 8):
        arch = f"efficientnet-b{u}"
        best_checkpoint = None
        best_checkpoint_auc = 0
        for checkpoint in os.listdir(run_folder + arch):
            auc = float(re.findall(r"(\d\.\d+(?=\.))", checkpoint)[0])
            if auc > best_checkpoint_auc:
                best_checkpoint = checkpoint
                best_checkpoint_auc = auc
        print(f"For arch {arch} loading checkpoint {best_checkpoint}")
        model = BigModel(train_df, test_df, IMAGE_DIR_TRAINING, IMAGE_DIR_TEST, arch)
        model = model.load_from_checkpoint(
            run_folder + arch + "/" + best_checkpoint,
            train_df=train_df,
            test_df=test_df,
            image_dir_training=IMAGE_DIR_TRAINING,
            image_dir_test=IMAGE_DIR_TEST,
            arch=arch,
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
    predictions[["image_name", "target"]].round(0).to_csv(
        "runs/ensemble_prediction_mean.csv", index=False
    )


if __name__ == "__main__":
    main()
