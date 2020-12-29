from __future__ import print_function
import argparse
import shutil
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from dla import DLA

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(args, model, device, train_loader, optimizer, epoch, loss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Flatt the image into 1D tensor
        # Training
        optimizer.zero_grad()
        output = model(data)
        output = loss(output, target)
        output.backward()
        optimizer.step()
        if batch_idx % 200 == 0 or batch_idx==len(train_loader):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    output,
                    # loss,
                ),
                end="\r",
                flush=True,
            )


def test(model, device, test_loader, loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Flatt the image into 1D tensor
            output = model(data)
            test_loss += loss(output, target)  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc,
        )
    )
    return acc


def save_ckp(state, checkpoint_dir):
    f_path = "cifar-best-checkpoint.pt"
    torch.save(state, f_path)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST RNA LAB")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        metavar="N",
        help="Number of neurons per layer",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=False,
        help="Path of checkpoint to restore, if none will start training from 0",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # transform = transforms.Compose([transforms.ToTensor()])
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset1 = torchvision.datasets.CIFAR10(
        ".data", train=True, download=True, transform=train_transforms
    )
    dataset2 = torchvision.datasets.CIFAR10(
        ".data", train=False, download=True, transform=test_transforms
    )

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # model = DPN92().to(device)
    model = DLA().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    epoch = 1
    loss = nn.CrossEntropyLoss()
    if args.load_checkpoint:
        print("Loading checkpoint args.load_checkpoint")
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
      Q  epoch = checkpoint["epoch"]
    best_acc = 0
    for epoch in range(epoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, loss)
        test_acc = test(model, device, test_loader, loss)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_acc > 95.0:
            print("Error < 5.0 achieved, stopped training")
            break
        if args.save_model and test_acc >= best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            print("Saving checkpoint as best model to cifar-best-checkpoint.pt")
            save_ckp(checkpoint, "")

        scheduler.step()


if __name__ == "__main__":
    main()
