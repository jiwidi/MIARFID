import numpy
import pandas as pd
import PIL.Image as Image
import torch
import torch.utils.data as pytorch_data
from torchvision import transforms
import time

class SIIMDataset(pytorch_data.Dataset):
    def __init__(self, df, transform, image_dir, test=False, use_metadata=False, include_2019 = False):
        self.df = df
        self.transform = transform
        self.test = test
        self.image_dir = image_dir
        self.use_metadata = use_metadata
        self.include_2019 = include_2019

        if self.use_metadata:
            # Transform dataframe
            dummies = pd.get_dummies(
                self.df["anatom_site_general_challenge"],
                dummy_na=True,
                dtype=numpy.uint8,
                prefix="site",
            )

            self.df = pd.concat([self.df, dummies.iloc[: self.df.shape[0]]], axis=1)
            self.df["sex"] = self.df["sex"].map({"male": 1, "female": 0})
            self.df["age_approx"] /= self.df["age_approx"].max()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        start = time.time()
        meta = self.df.iloc[idx]
        image_fn = (
            meta["image_name"] + ".jpg"
        )  # Use this when training with original images
        #         image_fn = meta['image_name'] + '.png'
        if self.test:
            img = Image.open(str(self.image_dir / "test") + "/" + image_fn).convert("RGB")
            #img = Image.open(self.image_dir + "/test" + "/" + image_fn).convert("RGB")
        else:
            if self.include_2019 and meta['patient_id'].startswith('IP_2019_'):
                img = Image.open(str(self.image_dir / "2019") + "/" + image_fn).convert("RGB")
                #img = Image.open(self.image_dir + "/2019" + "/" + image_fn).convert("RGB")
            else:
                img = Image.open(str(self.image_dir / "train") + "/" + image_fn).convert("RGB")
                #img = Image.open(self.image_dir + "/train" + "/" + image_fn).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.include_2019:
            # Now target will be a vector of size 9
            target = meta[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].tolist()
        else:
            target = meta['target']

        if self.use_metadata:
            metadata = ["sex", "age_approx"] + [
                col for col in meta.index if "site_" in col
            ]
            metadata.remove("anatom_site_general_challenge")
            metadata = numpy.array(meta[metadata], dtype=numpy.float64)
            # print(type(img) ,type(metadata.values), type(meta["target"]))
            return img, torch.from_numpy(metadata), torch.Tensor(target).long()
        #
        # print(time.time() - start)
        if self.test:
            return img
        return img, target


if __name__ == "__main__":
    train_df = pd.read_csv("data/train_full.csv")
    train_dataset = SIIMDataset(train_df, None, image_dir='data', use_metadata=True, include_2019=True)
    for img, met, y in train_dataset:
        print(img)
        print(met)
        print(y)
        break

