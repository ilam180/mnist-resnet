import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(None, None)  # training data
dataloader = DataLoader(dataset, batch_size=None, shuffle=True)  # choose batch size

criterion = nn.CrossEntropyLoss()  # or whatever we choose
model = None  # resnet

optimizer = optim.SGD(model.parameters(), lr=None)  # choose learning rate / optimizer

epochs = None  # choose epochs

for epoch in range(epochs):
    for data in dataloader:
        feature, target = data
        prediction = model(feature)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
