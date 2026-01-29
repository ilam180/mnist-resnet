import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .evaluate import evaluate


def train(model, trainloader, validloader, lr, epochs):

    criterion = nn.CrossEntropyLoss()  # or whatever we choose

    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()

    train_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(epochs):
        training_loss = 0
        for data in trainloader:
            feature, target = data
            prediction = model(feature)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()

        epoch_training_loss = training_loss / len(trainloader)
        train_losses.append(epoch_training_loss)

        validation_loss, validation_accuracy = evaluate(model, validloader, criterion)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

    return train_losses, validation_losses, validation_accuracies
