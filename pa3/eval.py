#!/usr/bin/env -S python3 -u

import os
import utils
import torch
import torch.nn.functional as F

from torchvision import datasets, transforms


def test_mnist(
    model: utils.LeNet,
    device,
    test_loader: torch.utils.data.DataLoader,
):
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
    print(
        "\nMNIST evaluation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
        .format(test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


def test_cifar10(
    model: utils.Net,
    device,
    test_loader: torch.utils.data.DataLoader,
):
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
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


def main():
    mnist_weights = os.path.join("./weights", "lenet-mnist.pt")
    cifar_weights = os.path.join("./weights", "cnn-cifar10.pt")

    device = torch.device("cuda")
    transform = transforms.ToTensor()
    test_kwargs = {
        "batch_size": 64,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }

    # Test on MNIST (LeNet)
    mnist_net = utils.LeNet()
    if os.path.exists(mnist_weights):
        mnist_net.load_state_dict(torch.load(mnist_weights))
        mnist_net = mnist_net.to(device)
        testing_set = datasets.MNIST(
            './datasets', train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testing_set, **test_kwargs)
        test_mnist(mnist_net, device, test_loader)

    # Test on CIFAR10 (Net)
    cifar_net = utils.Net()
    if os.path.exists(cifar_weights):
        cifar_net.load_state_dict(torch.load(cifar_weights))
        cifar_net = cifar_net.to(device)
        testing_set = datasets.CIFAR10(
            './datasets/cifar10', train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testing_set, **test_kwargs)
        test_cifar10(cifar_net, device, test_loader)


if __name__ == "__main__":
    main()

# Author: Blurgy <gy@blurgy.xyz>
# Date:   Jan 15 2021, 20:53 [CST]
