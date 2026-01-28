import torch
from data import dataset
from models import resnet
from train.train import train
from train.evaluate import evaluate

#run model goes here
if __name__ == "__main__":
    img_dir = 'images'
    annotations_file = 'annotations.csv'
    batch_size = 32
    lr = 0.001
    epochs = 10

    transform = dataset.transforms.Compose([
        dataset.transforms.Resize((224, 224)),
        dataset.transforms.ToTensor(),
    ])

    train_loader = dataset.create_dataloader(img_dir, annotations_file, batch_size, shuffle=True, transform=transform)
    valid_loader = dataset.create_dataloader(img_dir, annotations_file, batch_size, shuffle=False, transform=transform)

    model = resnet.ResNet18(num_classes=10)  #adjust classes
    train_losses, val_losses, val_accuracies = train(
        model,
        train_loader,
        valid_loader,
        batch_size,
        lr,
        epochs
    )

    #evaluate results
    final_val_loss, final_val_accuracy = evaluate(
        model,
        valid_loader,
        batch_size,
        criterion=torch.nn.CrossEntropyLoss()
    )

    print(f'Final Validation Loss: {final_val_loss}, Final Validation Accuracy: {final_val_accuracy}')