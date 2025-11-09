import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
from pathlib import Path



def load_images():
    image_dir = Path(__file__).parent / "images" / "images"
    images = []

    for image_path in image_dir.glob("*.png"):
        
        img = cv2.imread(str(image_path))

        print(f"{image_path.name}: shape={img.shape}")  
        
        images.append((image_path.name, img))

    return images

def macenko_rgb(image_path):
    img = cv2.imread(image_path)
    img = img.astype(np.float32)
    img[img == 0] = 1
    img_od = -np.log(255 / img)
    reshape_od = img_od.reshape((-1, 3))
    pca = PCA(n_components=2)
    pca.fit(reshape_od)
    stain_m = pca.components_.T
    stains = np.dot(reshape_od, np.linalg.pinv(stain_m).T)
    target_means = np.array([0.5, 0.5])
    target_stds = np.array([0.2, 0.2])
    stains_normalized = (stains - np.mean(stains, axis=(0, 1))) / np.std(stains, axis=(0, 1))
    stains_normalized = stains_normalized * target_stds + target_means
    od_normalized = np.dot(stains_normalized.reshape((-1, 2)), stain_m.T)
    od_normalized = od_normalized.reshape(img.shape)
    img_reconstructed = np.exp(-od_normalized)
    img_reconstructed = np.clip(img_reconstructed, 0, 1)
    img_reconstructed = (img_reconstructed * 255).astype(np.uint8)


def min_max_rgb(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_min = img_rgb.min(axis=(0, 1), keepdims=True)
    img_max = img_rgb.max(axis=(0, 1), keepdims=True)
    return (img_min, img_max)

if __name__ == "__main__":
    pass

