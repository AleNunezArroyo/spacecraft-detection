import cv2
import os
from glob import glob

# Ruta a las imágenes y etiquetas
image_dir = "spacecraft-dataset/images/train"
label_dir = "spacecraft-dataset/labels/train"

# Obtener todas las imágenes
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))

current_idx = 0  # Índice actual

while True:
    image_path = image_paths[current_idx]
    file_name = os.path.basename(image_path)
    label_path = os.path.join(label_dir, file_name.replace(".png", ".txt"))

    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo cargar: {image_path}")
        break

    h, w = img.shape[:2]

    # Leer archivo YOLO si existe
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, box_w, box_h = map(float, parts)

                # Convertir a coordenadas absolutas
                x = int((x_center - box_w / 2) * w)
                y = int((y_center - box_h / 2) * h)
                box_w = int(box_w * w)
                box_h = int(box_h * h)

                # Dibujar el rectángulo
                cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                cv2.putText(img, f"Class {int(class_id)}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("YOLO Annotation", img)
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key == 81:  # Flecha izquierda
        current_idx = (current_idx - 1) % len(image_paths)
    elif key == 83:  # Flecha derecha
        current_idx = (current_idx + 1) % len(image_paths)

cv2.destroyAllWindows()