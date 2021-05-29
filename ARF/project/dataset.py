import numpy
import pandas as pd
import PIL.Image as Image
import torch
import torch.utils.data as pytorch_data
from torchvision import transforms
import time
import albumentations as A

class SIIMDataset(pytorch_data.Dataset):
    def __init__(self, df, transform, image_dir, test=False, use_metadata=False, include_2019 = False, use_9_classes=False):
        self.df = df
        self.transform = transform
        self.test = test
        self.image_dir = image_dir
        self.use_metadata = use_metadata
        self.include_2019 = include_2019
        self.use_9_classes = use_9_classes

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
            
            if self.test:
                self.df['site_anterior torso'] = [0 for i in range(len(self.df))]
                self.df['site_lateral torso'] = [0 for i in range(len(self.df))]
                self.df['site_posterior torso'] = [0 for i in range(len(self.df))]
                self.df = self.df[['image_name', 'patient_id', 'sex', 'age_approx',
                    'anatom_site_general_challenge', 'site_anterior torso', 'site_head/neck',
                    'site_lateral torso', 'site_lower extremity', 'site_oral/genital',
                    'site_palms/soles', 'site_posterior torso', 'site_torso',
                    'site_upper extremity', 'site_nan'
                ]]


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
            img = numpy.asarray(Image.open(str(self.image_dir / "test") + "/" + image_fn).convert("RGB"))
            #img = numpy.array(Image.open(self.image_dir + "/test" + "/" + image_fn).convert("RGB"))
        else:
            if self.include_2019 and meta['patient_id'].startswith('IP_2019_'):
                    img = numpy.asarray(Image.open(str(self.image_dir / "2019") + "/" + image_fn).convert("RGB"))
                    #img = numpy.array(Image.open(self.image_dir + "/2019" + "/" + image_fn).convert("RGB"))
            else:
                img = numpy.asarray(Image.open(str(self.image_dir / "train") + "/" + image_fn).convert("RGB"))
                #img = numpy.array(Image.open(self.image_dir + "/train" + "/" + image_fn).convert("RGB"))

        if self.transform is not None:
            img = self.transform(image=img)
            img = img['image'].astype(numpy.float32)
            img = numpy.moveaxis(img, -1, 0)            # Convert to channels first

        if not self.test:
            if self.include_2019:
                if self.use_metadata or self.use_9_classes:
                    # Now target will be a vector of size 9
                    target = meta[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].tolist()
                else:
                    target = meta['target']
            else:
                target = meta['target']

        if self.use_metadata:
            metadata = ["sex", "age_approx"] + [
                col for col in meta.index if "site_" in col
            ]
            metadata.remove("anatom_site_general_challenge")
            metadata = numpy.array(meta[metadata], dtype=numpy.float64)
            # print(type(img) ,type(metadata.values), type(meta["target"])
            if self.test:
                return img, torch.from_numpy(metadata)
            else:
                return img, torch.from_numpy(metadata), torch.Tensor(target).long()
        
        if self.use_9_classes:
            return img, torch.Tensor(target).long()
        #
        # print(time.time() - start)
        return img, target


if __name__ == "__main__":

    transform_test = A.Compose(
            [
                A.Resize(
                    224,
                    224
                ),  # Use this when training with original images
                A.Normalize()
            ]
        )
    train_df = pd.read_csv("data/train_full.csv")
    train_dataset = SIIMDataset(train_df, transform=transform_test, image_dir='data', use_metadata=True, include_2019=True)
    print(train_dataset[0][0])
    test_df = pd.read_csv("data/test_full.csv")
    test_dataset = SIIMDataset(test_df, transform=transform_test, image_dir='data', use_metadata=True, test=True, include_2019=True)
    print(test_dataset[0][0])

