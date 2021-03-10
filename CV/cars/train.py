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
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

writer = SummaryWriter()
from bcnn import BilinearModel
from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(model, dataloader, optimizer, scheduler, loss_fn, epoch):
    # Set the model into train mode
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    datacount = len(dataloader)

    for batch_idx, (train_batch, labels_batch) in enumerate(dataloader):
        # move the data onto the device
        train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
        optimizer.zero_grad()

        # compute model outputs and loss
        outputs = model(train_batch)
        loss = loss_fn(outputs, labels_batch.squeeze())
        loss.backward()

        # after computing gradients based on current batch loss,
        # apply them to parameters
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch.squeeze()).sum().item()

        # write to tensorboard
        writer.add_scalar(
            "train/loss",
            train_loss / (batch_idx + 1),
            (datacount * (epoch + 1)) + (batch_idx + 1),
        )
        writer.add_scalar(
            "train/accuracy",
            100.0 * correct / total,
            (datacount * (epoch + 1)) + (batch_idx + 1),
        )
        writer.add_scalar(
            "train/lr",
            scheduler._last_lr[0],
            (datacount * (epoch + 1)) + (batch_idx + 1),
        )
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(train_batch),
                len(dataloader.dataset),
                100.0 * batch_idx / len(dataloader),
                (train_loss / (batch_idx + 1)),
                # loss,
            ),
            end="\r",
            flush=True,
        )
    print()
    return train_loss / datacount, 100.0 * correct / total


def test(model, dataloader, loss_fn, epoch):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    datacount = len(dataloader)

    with torch.no_grad():
        for batch_idx, (test_batch, labels_batch) in enumerate(dataloader):

            # move the data onto device
            test_batch, labels_batch = test_batch.to(device), labels_batch.to(device)

            # compute the model output
            outputs = model(test_batch)
            loss = loss_fn(outputs, labels_batch.squeeze())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch.squeeze()).sum().item()
            # log the test_loss
            writer.add_scalar(
                "test/loss",
                test_loss / (batch_idx + 1),
                (datacount * (epoch + 1)) + (batch_idx + 1),
            )
            writer.add_scalar(
                "test/accuracy",
                100.0 * correct / total,
                (datacount * (epoch + 1)) + (batch_idx + 1),
            )

    test_loss = test_loss / datacount
    acc = 100 * correct / total
    print("Test accuracy:", acc)
    return test_loss, acc


def save_ckp(state, checkpoint_dir):
    f_path = "cars-best-checkpoint.pt"
    torch.save(state, f_path)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch cars CV LAB")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
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
        cuda_kwargs = {"num_workers": 8, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

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
    # Load
    x_train = np.load("data/x_train.npy")
    x_test = np.load("data/x_test.npy")

    x_train = torch.from_numpy(x_train).squeeze().permute(0, 3, 1, 2).float()
    x_test = torch.from_numpy(x_test).squeeze().permute(0, 3, 1, 2).float()

    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    # Fix class labels
    y_train = y_train - 1
    y_test = y_test - 1

    y_train = torch.from_numpy(y_train).squeeze().long()
    y_test = torch.from_numpy(y_test).squeeze().long()

    print("X shape: ", x_train.shape)
    print("Y shape: ", y_train.shape)
    dataset1 = torch.utils.data.TensorDataset(x_train, y_train.unsqueeze(1))
    dataset2 = torch.utils.data.TensorDataset(x_test, y_test.unsqueeze(1))

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = BilinearModel().to(device)
    # print(summary(model, (3, 100, 100)))

    print(
        "Trainable parameters",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.05, steps_per_epoch=len(train_loader), epochs=args.epochs
    )
    epoch = 1
    loss = nn.CrossEntropyLoss()
    if args.load_checkpoint:
        print("Loading checkpoint args.load_checkpoint")
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
    best_acc = 0

    l_train_loss = []
    l_test_loss = []
    l_train_acc = []
    l_test_acc = []
    l_lr = []
    for epoch in range(epoch, args.epochs + 1):
        train_loss, train_acc = train(
            model, train_loader, optimizer, scheduler, loss, epoch
        )
        test_loss, test_acc = test(model, test_loader, loss, epoch)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_acc > 65.0:
            print("Error < 35.0 achieved, stopped training")
            break
        if args.save_model and test_acc >= best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            print("Saving checkpoint as best model to cars-best-checkpoint.pt")
            save_ckp(checkpoint, "")

        l_train_loss.append(train_loss)
        l_test_loss.append(test_loss)
        l_train_acc.append(train_acc)
        l_test_acc.append(test_acc)
        l_lr.append(scheduler._last_lr[0])

    # PLOTS
    fig = plt.figure()
    plt.plot(l_train_loss, color="red", label="Train")
    plt.plot(l_test_loss, color="blue", label="Test")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=8)
    plt.legend()
    plt.grid()
    fig.savefig("figures/cars_loss.png")
    plt.close()

    fig = plt.figure()
    plt.plot(l_train_acc, color="red", label="Train")
    plt.plot(l_test_acc, color="blue", label="Test")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy", fontsize=8)
    plt.legend()
    plt.grid()
    fig.savefig("figures/cars_acc.png")
    plt.close()

    fig = plt.figure()
    plt.plot(l_lr, color="orange", label="Learning rate")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Learning rate", fontsize=8)
    plt.legend()
    plt.grid()
    fig.savefig("figures/cars_lr.png")
    plt.close()


if __name__ == "__main__":
    main()
