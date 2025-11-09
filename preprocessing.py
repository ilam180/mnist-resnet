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

if __name__ == "__main__":
    temp()
