# Classifying Colorectal Polyps through MNIST-ResNet18 Architecture
## Purdue Bioinformatics Club: Computer Vision Team
Machine Learning project for detecting potential colorectal cancer based on the ResNet18 architecture, implemented with Python/PyTorch. 

## Project Structure

```text
mnist-resnet/
│
├── data/               # Datasets (train, test, raw, processed)
│   ├── dataset.py            
│   └── transforms.py     
│
├── images/          # training images from MHIST dataset
│
├── models/                
│   └── resnet.py   # ResNet18
│
├── train/             # Saved models or checkpoints
│   ├── evaluate.py
│   ├── train.py        
│   └── utils.py
│
├── annotations.csv
├── preprocessing.py
└── README.md
```

