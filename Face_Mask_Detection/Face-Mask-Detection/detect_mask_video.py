"""
detect_mask_video.py
---------------------
Real-time face-mask detection from a webcam feed.

Pipeline:
    1. OpenCV's pre-trained DNN face detector (Caffe SSD ResNet-10) finds
       face bounding boxes in each frame.
    2. Each detected face is cropped, preprocessed, and passed through the
       trained mask_detector model (MobileNetV2 head) to classify
       Mask / No Mask.
    3. Boxes + labels + confidence are drawn on the frame.

Requires the face detector files (already provided in face_detector/):
    face_detector/deploy.prototxt
    face_detector/res10_300x300_ssd_iter_140000.caffemodel

Usage:
    python detect_mask_video.py --model mask_detector.model --confidence 0.5
Press 'q' to quit.
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
    ap.add_argument("-f", "--face", type=str, default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.keras",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak face detections")
    ap.add_argument("-s", "--source", type=int, default=0,
                    help="camera index (0 = default webcam)")
    return ap.parse_args()


def detect_and_predict_mask(frame, face_net, mask_net, min_confidence):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")
        start_x, start_y = max(0, start_x), max(0, start_y)
        end_x, end_y = min(w - 1, end_x), min(h - 1, end_y)

        face = frame[start_y:end_y, start_x:end_x]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((start_x, start_y, end_x, end_y))

    if faces:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return locs, preds


def main():
    args = parse_args()

    print("[INFO] loading face detector model...")
    prototxt_path = os.path.join(args.face, "deploy.prototxt")
    weights_path = os.path.join(args.face, "res10_300x300_ssd_iter_140000.caffemodel")
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)

    print("[INFO] loading face mask detector model...")
    mask_net = load_model(args.model)

    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args.source)

    while True:
        ret, frame = vs.read()
        if not ret:
            print("[WARN] failed to grab frame, exiting...")
            break

        frame = cv2.resize(frame, (720, int(frame.shape[0] * 720 / frame.shape[1])))
        locs, preds = detect_and_predict_mask(frame, face_net, mask_net, args.confidence)

        for (box, pred) in zip(locs, preds):
            (start_x, start_y, end_x, end_y) = box
            (mask, without_mask) = pred

            label = "Mask" if mask > without_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label_text = f"{label}: {max(mask, without_mask) * 100:.1f}%"

            cv2.putText(frame, label_text, (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        cv2.imshow("Face Mask Detector - press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()