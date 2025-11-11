import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
from pathlib import Path

def load_images():
    image_dir = Path(__file__).parent / "images"
    file = Path(__file__).parent / "image_files_info.txt"

    images = []

    with open(file, "w") as f:
        for image_path in image_dir.glob("*.png"):
            img = cv2.imread(str(image_path))

            # IQR range
            red = img[:, :, 2].flatten()
            green = img[:, :, 1].flatten()
            blue = img[:, :, 0].flatten()

            red_q1, red_q3 = np.percentile(red, [25,75])
            green_q1, green_q3 = np.percentile(green, [25,75])
            blue_q1, blue_q3 = np.percentile(blue, [25,75])
            
            red_iqr = red_q3 - red_q1
            green_iqr = green_q3 - green_q1
            blue_iqr = blue_q3 - blue_q1

            # finding the min and max values
            img_min, img_max = min_max_rgb(str(image_path))

            red_min = img_min[0, 0, 0]
            green_min = img_min[0, 0, 1]
            blue_min = img_min[0, 0, 2]

            red_max = img_max[0, 0, 0]
            green_max = img_max[0, 0, 1]
            blue_max = img_max[0, 0, 2]

            f.write(f"{image_path.name}: dimensions={img.shape[:2]} \n" 
                    f"\tRed: min={red_min}, max={red_max}, IQR={red_iqr:.2f}\n"
                    f"\tGreen: min={green_min}, max={green_max}, IQR={green_iqr:.2f}\n"
                    f"\tBlue: min={blue_min}, max={blue_max}, IQR={blue_iqr:.2f}\n\n"
                    )  
            
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

    ##hardcoded target means and stds. pick a target.
    target_means = np.array([0.5, 0.5])
    target_stds = np.array([0.2, 0.2])
    ##

    stains_normalized = (stains - np.mean(stains, axis=(0, 1))) / np.std(stains, axis=(0, 1))
    stains_normalized = stains_normalized * target_stds + target_means
    od_normalized = np.dot(stains_normalized.reshape((-1, 2)), stain_m.T)
    od_normalized = od_normalized.reshape(img.shape)
    img_reconstructed = np.exp(-od_normalized)
    img_reconstructed = np.clip(img_reconstructed, 0, 1)
    img_reconstructed = (img_reconstructed * 255).astype(np.uint8)
    return img_reconstructed


def min_max_rgb(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_min = img_rgb.min(axis=(0, 1), keepdims=True)
    img_max = img_rgb.max(axis=(0, 1), keepdims=True)
    return (img_min, img_max)

if __name__ == "__main__":
    load_images()
    #cv2.imwrite("test aaf.png", macenko_rgb("images/images/MHIST_aaf.png"))
