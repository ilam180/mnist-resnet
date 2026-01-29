import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def evaluate(model, validloader, criterion):

    model.eval()
    validation_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for feature, target in validloader:
            prediction = model(feature)
            loss = criterion(prediction, target)
            validation_loss += loss.item()

            _, predicted = torch.max(prediction, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    avg_loss = validation_loss / len(validloader)
    accuracy = correct / total

    return avg_loss, accuracy
