# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Torch
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from torchsummary import summary

from CNN import CNN


def batch_gd(model, criterion, train_loader, validation_loader, epochs):
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    for e in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            output = model(inputs)

            loss = criterion(output, targets)

            train_loss.append(loss.item())  # torch to numpy world

            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)

        validation_loss = []

        for inputs, targets in validation_loader:

            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)

            loss = criterion(output, targets)

            validation_loss.append(loss.item())  # torch to numpy world

        validation_loss = np.mean(validation_loss)

        train_losses[e] = train_loss
        validation_losses[e] = validation_loss

        dt = datetime.now() - t0

        print(
            f"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Test_loss:{validation_loss:.3f} Duration:{dt}"
        )

    return train_losses, validation_losses


def accuracy(loader):
    n_correct = 0
    n_total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]
    acc = n_correct / n_total
    return acc


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    dataset = datasets.ImageFolder("Dataset", transform=transform)

    indices = list(range(len(dataset)))
    split = int(np.floor(0.85 * len(dataset)))  # training size
    validation = int(np.floor(0.8 * split))

    print()
    print(f"length of train size:\t\t{validation}")
    print(f"length of validation size:\t{split - validation}")
    print(f"length of test size:\t\t{len(dataset) - validation}")

    np.random.shuffle(indices)

    train_indices, validation_indices, test_indices = (
        indices[:validation],
        indices[validation:split],
        indices[split:],
    )

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    targets_size = len(dataset.class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = CNN(targets_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # this include softmax + cross entropy loss
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=validation_sampler
    )

    train_acc = accuracy(train_loader)
    test_acc = accuracy(test_loader)
    validation_acc = accuracy(validation_loader)
    print()
    print(f"Train accuracy:\t\t{train_acc}")
    print(f"Test accuracy:\t{test_acc}")
    print(f"Validation accuracy:\t\t{validation_acc}")


    train_losses, validation_losses = batch_gd(
        model, criterion, train_loader, validation_loader, 5
    )

    torch.save(model.state_dict(), 'plant_disease_model_1_latest.pt')

