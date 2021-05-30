import pandas as pd
import os
from pathlib import Path


def main():
    dir = os.path.dirname("runs/predictions/ensemble")

    predictions = None
    for csv in os.listdir(dir):
        prediction = pd.read_csv(csv)
        if not isinstance(predictions, pd.DataFrame):
            predictions = prediction
        else:
            predictions["target-" + csv] = prediction.target

    # Ensemble
    aux = []
    for column in predictions.columns:
        if "target" in column:
            aux.append(column)
    print(f"Ensembling columns {aux}")
    predictions["target"] = predictions[aux].sum(axis=1) / len(aux)
    predictions[["image_name", "target"]].round(6).to_csv(
        "runs/ensemble2_prediction_mean.csv", index=False
    )


if __name__ == "__main__":
    main()
