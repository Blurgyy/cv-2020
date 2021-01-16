#!/usr/bin/env -S python -u

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(
# mode='Verbose', color_scheme='Linux', call_pdb=False)

import torch
import torch.nn.functional as F
import os

from torchvision import datasets, transforms

import utils


def train(
    model: utils.Net,
    device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    model.train()
    interval = 10
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_id % interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_id * len(data), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss.item()))


def test(
    model: utils.Net,
    device,
    test_loader: torch.utils.data.DataLoader,
) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    ret = 100. * correct / len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
          .format(test_loss, correct, len(test_loader.dataset), ret))
    return ret


def main():
    torch.manual_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_kwargs = {"batch_size": 64}
    test_kwargs = {"batch_size": 64}
    if device == torch.device("cuda"):
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.ToTensor()
    training_set = utils.CIFAR11(
        './datasets/cifar11', train=True, transform=transform, download=True)
    testing_set = utils.CIFAR11(
        './datasets/cifar11', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(training_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testing_set, **test_kwargs)

    model = utils.Net().to(device)

    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=1,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.7,
    )

    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    max_acc = 0
    for epoch in range(0, 16):
        train(model, device, train_loader, optimizer, epoch + 1)
        cur_acc = test(model, device, test_loader)
        if max_acc < cur_acc:
            max_acc = cur_acc
            torch.save(model.state_dict(), "./weights/cnn-cifar11.pt")
        scheduler.step()


if __name__ == "__main__":
    main()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 09 2021, 14:50 [CST]
