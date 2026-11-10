"""
detect_mask_image.py
----------------------
Run mask detection on a single static image.

Usage:
    python detect_mask_image.py --image path/to/photo.jpg --model mask_detector.model
Output image (with boxes/labels drawn) is saved next to the input as
    <name>_annotated.jpg
"""

import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-f", "--face", type=str, default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.keras",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak face detections")
    return ap.parse_args()


def main():
    args = parse_args()

    print("[INFO] loading models...")
    prototxt_path = os.path.join(args.face, "deploy.prototxt")
    weights_path = os.path.join(args.face, "res10_300x300_ssd_iter_140000.caffemodel")
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)
    mask_net = load_model(args.model)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {args.image}")
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < args.confidence:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")
        start_x, start_y = max(0, start_x), max(0, start_y)
        end_x, end_y = min(w - 1, end_x), min(h - 1, end_y)

        face = image[start_y:end_y, start_x:end_x]
        if face.size == 0:
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, without_mask) = mask_net.predict(face)[0]
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label_text = f"{label}: {max(mask, without_mask) * 100:.1f}%"

        cv2.putText(image, label_text, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)

    out_path = os.path.splitext(args.image)[0] + "_annotated.jpg"
    cv2.imwrite(out_path, image)
    print(f"[INFO] annotated image saved to {out_path}")


if __name__ == "__main__":
    main()