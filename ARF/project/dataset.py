import PIL.Image as Image
import torch.utils.data as pytorch_data
import numpy
from torchvision import transforms


class SIIMDataset(pytorch_data.Dataset):
    def __init__(
        self, df, transform, image_dir, test=False,
    ):
        self.df = df
        self.transform = transform
        self.test = test
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        meta = self.df.iloc[idx]
        image_fn = (
            meta["image_name"] + ".jpg"
        )  # Use this when training with original images
        #         image_fn = meta['image_name'] + '.png'
        if self.test:
            img = Image.open(str(self.image_dir / ("test/" + image_fn))).convert("RGB")
        else:
            img = Image.open(str(self.image_dir / ("train/" + image_fn))).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, meta["target"]
