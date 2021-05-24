import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.metrics.classification import AUROC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from dataset import SIIMDataset

lr = 0.00005
max_epochs = 50
batch_size = 8
num_workers = os.cpu_count()
label_smoothing = 0.03
pos_weight = 3.2


class BigModel(pl.LightningModule):
    def __init__(self, train_df, test_df, image_dir, arch):
        super().__init__()
        self.arch = arch
        self.net = EfficientNet.from_pretrained(arch, advprop=True)
        self.net._fc = torch.nn.Linear(
            in_features=self.net._fc.in_features, out_features=1, bias=True
        )

        self.train_df = train_df
        self.test_df = test_df
        self.image_dir = image_dir

        # Split
        patient_means = train_df.groupby(["patient_id"])["target"].mean()
        patient_ids = train_df["patient_id"].unique()
        train_idx, val_idx = train_test_split(
            np.arange(len(patient_ids)), test_size=0.2
        )

        self.pid_train = patient_ids[train_idx]
        self.pid_val = patient_ids[val_idx]

        # Transforms
        self.transform_train = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),  # Use this when training with original images
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),  # Use this when training with original images
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        return [optimizer]

    def step(self, batch):
        # return batch loss
        x, y = batch
        y_hat = self(x).flatten()
        y_smo = y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y_smo.type_as(y_hat), pos_weight=torch.tensor(pos_weight)
        )
        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_nb):
        # hardware agnostic training
        loss, y, y_hat = self.step(batch)
        acc = (y_hat.round() == y).float().mean().item()
        self.log("acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {"val_loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs])
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        auc = (
            AUROC()(preds=y_hat, target=y) if y.float().mean() > 0 else 0.5
        )  # skip sanity check
        acc = (y_hat.round() == y).float().mean().item()
        print(f"Epoch {self.current_epoch} acc:{acc} auc:{auc}")
        self.log("avg_val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"avg_val_loss": avg_loss, "val_auc": auc, "val_acc": acc}

    def test_step(self, batch, batch_nb):
        x = batch
        y_hat = self(x).flatten().sigmoid()
        return {"y_hat": y_hat}

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        self.test_df["target"] = y_hat.tolist()
        self.test_df[["image_name", "target"]].to_csv(
            f"runs/predictions/submission_{self.arch}.csv", index=False
        )

    def train_dataloader(self):
        ds_train = SIIMDataset(
            self.train_df[self.train_df["patient_id"].isin(self.pid_train)],
            self.transform_train,
            self.image_dir,
        )

        classes = self.train_df[self.train_df["patient_id"].isin(self.pid_train)][
            "target"
        ].to_numpy()

        class_sample_count = np.array(
            [len(np.where(classes == t)[0]) for t in np.unique(classes)]
        )
        weight = 1.0 / class_sample_count

        samples_weight = np.array([weight[t] for t in classes])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return DataLoader(
            ds_train,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        ds_val = SIIMDataset(
            self.train_df[self.train_df["patient_id"].isin(self.pid_val)],
            self.transform_test,
            self.image_dir,
        )
        return DataLoader(
            ds_val,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        ds_test = SIIMDataset(
            self.test_df, self.transform_test, self.image_dir, test=True
        )
        return DataLoader(
            ds_test,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=False,
        )


class Model2Branches(pl.LightningModule):
    def __init__(self, train_df, test_df, image_dir, arch, n_meta_features):
        super().__init__()
        self.net = EfficientNet.from_pretrained(arch, advprop=True)
        self.net._fc = torch.nn.Linear(
            in_features=self.net._fc.in_features, out_features=500, bias=True
        )

        self.meta = nn.Sequential(
            nn.Linear(n_meta_features, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.output = nn.Linear(500 + 250, 1)

        self.train_df = train_df
        self.test_df = test_df
        self.image_dir = image_dir

        # Split
        patient_means = train_df.groupby(["patient_id"])["target"].mean()
        patient_ids = train_df["patient_id"].unique()
        train_idx, val_idx = train_test_split(
            np.arange(len(patient_ids)), test_size=0.2
        )

        self.pid_train = patient_ids[train_idx]
        self.pid_val = patient_ids[val_idx]

        # Transforms
        self.transform_train = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),  # Use this when training with original images
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),  # Use this when training with original images
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, inputs):
        x, metadata = inputs
        cnn_output = self.net(x)
        meta_output = self.meta(metadata.float())
        concat = torch.cat((cnn_output, meta_output), dim=1)
        output = self.output(concat)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        return [optimizer]

    def step(self, batch):
        # return batch loss
        x, metadata, y = batch
        y_hat = self((x, metadata)).flatten()
        y_smo = y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y_smo.type_as(y_hat), pos_weight=torch.tensor(pos_weight)
        )
        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_nb):
        # hardware agnostic training
        loss, y, y_hat = self.step(batch)
        acc = (y_hat.round() == y).float().mean().item()
        self.log("acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {"val_loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs])
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        auc = (
            AUROC()(preds=y_hat, target=y) if y.float().mean() > 0 else 0.5
        )  # skip sanity check
        acc = (y_hat.round() == y).float().mean().item()
        print(f"Epoch {self.current_epoch} acc:{acc} auc:{auc}")
        self.log("avg_val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_nb):
        x = batch
        y_hat = self(x).flatten().sigmoid()
        return {"y_hat": y_hat}

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        df_test["target"] = y_hat.tolist()
        N = len(glob("submission*.csv"))
        df_test.target.to_csv(f"submission{N}.csv")
        return {"tta": N}

    def train_dataloader(self):
        ds_train = SIIMDataset(
            self.train_df[self.train_df["patient_id"].isin(self.pid_train)],
            self.transform_train,
            self.image_dir,
            use_metadata=True,
        )

        classes = self.train_df[self.train_df["patient_id"].isin(self.pid_train)][
            "target"
        ].to_numpy()

        class_sample_count = np.array(
            [len(np.where(classes == t)[0]) for t in np.unique(classes)]
        )
        weight = 1.0 / class_sample_count

        samples_weight = np.array([weight[t] for t in classes])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return DataLoader(
            ds_train,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        ds_val = SIIMDataset(
            self.train_df[self.train_df["patient_id"].isin(self.pid_val)],
            self.transform_test,
            self.image_dir,
            use_metadata=True,
        )
        return DataLoader(
            ds_val,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        ds_test = SIIMDataset(self.test_df, self.transform_test, self.image_dir,)
        return DataLoader(
            ds_test,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=False,
        )

