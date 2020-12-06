from __future__ import print_function
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fl1 = nn.Linear(784, 2048)
#         self.bn1 = nn.BatchNorm1d(2048)
#         self.relu1 = nn.ReLU()
#         self.fl2 = nn.Linear(2048, 2048)
#         self.bn2 = nn.BatchNorm1d(2048)
#         self.relu2 = nn.ReLU()
#         self.fl3 = nn.Linear(2048, 2048)
#         self.bn3 = nn.BatchNorm1d(2048)
#         self.relu3 = nn.ReLU()
#         self.fl4 = nn.Linear(2048, 10)

#     def forward(self, x):
#         x = self.relu1(self.bn1(self.fl1(x)))
#         x = self.relu2(self.bn2(self.fl2(x)))
#         x = self.relu3(self.bn3(self.fl3(x)))
#         x = self.fl4(x)
#         return x


class noisylayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.d_model, args.d_model)
        self.bn = nn.BatchNorm1d(args.d_model)
        self.gn = GaussianNoise(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.gn(self.bn(self.linear(x))))
        return out


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.
    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device).float()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


# Creating our Neural Network - Fully Connected
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.gn0 = GaussianNoise(0.1)
        self.linear = nn.Linear(784, args.d_model)
        self.bn = nn.BatchNorm1d(args.d_model)
        self.gn = GaussianNoise(0.1)
        self.relu = nn.ReLU()
        self.sequential = nn.Sequential(*(noisylayer(args) for i in range(1)))
        self.classifier = nn.Linear(args.d_model, 10)

    def forward(self, x):
        out = self.gn0(x)
        out = self.relu(self.gn(self.bn(self.linear(out))))
        out = self.sequential(out)
        out = self.classifier(out)
        return out


def train(args, model, device, train_loader, optimizer, epoch, loss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Flatt the image into 1D tensor
        data = data.flatten(start_dim=1)
        # Training
        optimizer.zero_grad()
        output = model(data)
        output = loss(output, target)
        output.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
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
            data = data.flatten(start_dim=1)
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
    f_path = "mnist-best-checkpoint.pt"
    torch.save(state, f_path)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST RNA LAB")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(p=0.0),
            torchvision.transforms.RandomAffine(degrees=3, translate=(0.1, 0.1)),
            torchvision.transforms.ToTensor(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    dataset1 = datasets.MNIST(
        ".data", train=True, download=True, transform=train_transforms
    )
    dataset2 = datasets.MNIST(".data", train=False, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epoch = 1
    # Learning Rate Annealing (LRA) scheduling
    # lr = 0.1     if epoch < 25
    # lr = 0.01    if 30 <= epoch < 50
    # lr = 0.001   if epoch >= 50
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 125], gamma=0.1
    )
    if args.load_checkpoint:
        print("Loading checkpoint args.load_checkpoint")
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
    best_acc = 0
    for epoch in range(epoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        test_acc = test(model, device, test_loader, criterion)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_acc > 99.2:
            print("Error < 0.8 achieved, stopped training")
            break
        if args.save_model and test_acc >= best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            print("Saving checkpoint as best model to mnist-best-checkpoint.pt")
            save_ckp(checkpoint, "")
        scheduler.step()


if __name__ == "__main__":
    main()
