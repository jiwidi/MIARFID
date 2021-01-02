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

writer = SummaryWriter()
from resnet import ResNet18

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

        # # convert to torch Variables
        # train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        # clear the previous grad
        optimizer.zero_grad()

        # compute model outputs and loss
        outputs = model(train_batch)
        loss = loss_fn(outputs, labels_batch)
        loss.backward()

        # after computing gradients based on current batch loss,
        # apply them to parameters
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()
        # get learning rate

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
            loss = loss_fn(outputs, labels_batch)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

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
    acc = 100.0 * correct / total
    print("Test accuracy:", acc)
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
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
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
        cuda_kwargs = {"num_workers": 8, "pin_memory": True, "shuffle": True}
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
    model = ResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=200
    )  # epoch 187
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

    for epoch in range(epoch, args.epochs + 1):
        train(model, train_loader, optimizer, scheduler, loss, epoch)
        test_acc = test(model, test_loader, loss, epoch)
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


if __name__ == "__main__":
    main()
