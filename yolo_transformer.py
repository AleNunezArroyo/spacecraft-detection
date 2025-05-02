import os
import json
from pathlib import Path
from PIL import Image

# Ruta base del dataset
BASE_PATH = os.path.join(os.getcwd(), "spacecraft-dataset")
IMAGES_PATH = os.path.join(BASE_PATH, "images")
LABELS_PATH = os.path.join(BASE_PATH, "labels")

# Cargar anotaciones desde el txt original
with open(os.path.join(BASE_PATH, "all_bbox.txt"), "r") as f:
    original_data = json.load(f)

# Función para convertir y guardar en formato YOLO
def convert_to_yolo(image_path, bboxes, label_path):
    with Image.open(image_path) as img:
        width, height = img.size

    with open(label_path, "w") as f:
        for bbox in bboxes:
            max_x, max_y, min_x, min_y = bbox
            x = min_x
            y = min_y
            w = max_x - min_x
            h = max_y - min_y

            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# Procesar imágenes por conjunto
def process_split(split_name):
    img_dir = os.path.join(IMAGES_PATH, split_name)
    label_dir = os.path.join(LABELS_PATH, split_name)
    os.makedirs(label_dir, exist_ok=True)

    for img_path in Path(img_dir).glob("*.png"):
        file_name = img_path.name
        img_id_str = file_name.split("_")[-1].split(".")[0]
        bboxes = original_data.get(img_id_str, [])

        label_path = os.path.join(label_dir, file_name.replace(".png", ".txt"))
        convert_to_yolo(img_path, bboxes, label_path)

# Ejecutar para cada conjunto
for split in ["train", "val", "test"]:
    process_split(split)

print("Archivos YOLO generados")
