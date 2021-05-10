from __future__ import print_function

import argparse
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from utils import *


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style_img = image_loader(args.style_img)
    content_img = image_loader(args.content_img)
    input_img = content_img.clone()
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Additionally, VGG networks are trained on images with each channel
    # normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We will use them to normalize the image before sending it into the network.
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=args.steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
    )

    plt.figure()
    image = output.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    plt.savefig("resultstyle.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--style_img",
        default="images/style.png",
        type=str,
        help="Style image to use in transfer style",
    )
    parser.add_argument(
        "--content_img",
        default="images/upv.png",
        type=str,
        help="Content image to use in transfer style",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Content image to use in transfer style",
    )
    parser.add_argument(
        "--content_weight",
        type=int,
        default=1,
        help="Content image to use in transfer style",
    )
    parser.add_argument(
        "--style_weight",
        type=int,
        default=1000000,
        help="Content image to use in transfer style",
    )

    args = parser.parse_args()
    main(args)
