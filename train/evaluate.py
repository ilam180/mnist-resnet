import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def evaluate(model, X_valid, y_valid, batch_size, criterion):
    dataset = TensorDataset(X_valid, y_valid)
    validationloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    validation_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for feature, target in validationloader:
            prediction = model(feature)
            loss = criterion(prediction, target)
            validation_loss += loss.item()

            _, predicted = torch.max(prediction, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    avg_loss = validation_loss / len(validationloader)
    accuracy = correct / total

    return avg_loss, accuracy
