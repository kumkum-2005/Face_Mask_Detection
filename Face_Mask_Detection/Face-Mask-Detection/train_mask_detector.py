"""
train_mask_detector.py
-----------------------
Trains a face-mask classifier using transfer learning on MobileNetV2.

Expected dataset layout (download e.g. the Kaggle "Face Mask Detection"
dataset and arrange it like this):

    dataset/
        with_mask/
            img1.jpg
            img2.jpg
            ...
        without_mask/
            img1.jpg
            img2.jpg
            ...

Usage:
    python train_mask_detector.py --dataset dataset --epochs 20 --plot outputs/training_plot.png

Output:
    mask_detector.model   -> trained Keras model (saved in .keras format)
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # so it works without a display (servers, CI, etc.)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset (with_mask/without_mask subfolders)")
    ap.add_argument("-p", "--plot", type=str, default="outputs/training_plot.png",
                    help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                    help="path to output trained model")
    ap.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
    ap.add_argument("-b", "--batch-size", type=int, default=32, help="training batch size")
    ap.add_argument("-l", "--lr", type=float, default=1e-4, help="initial learning rate")
    return ap.parse_args()


def load_dataset(dataset_path):
    print("[INFO] loading images...")
    image_paths = []
    for category in ("with_mask", "without_mask"):
        cat_dir = os.path.join(dataset_path, category)
        if not os.path.isdir(cat_dir):
            raise FileNotFoundError(
                f"Expected folder '{category}' inside {dataset_path}. "
                f"Check the README for the required dataset layout."
            )
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append((os.path.join(cat_dir, fname), category))

    data, labels = [], []
    for path, label in image_paths:
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    print(f"[INFO] loaded {len(data)} images "
          f"({np.sum(labels == 'with_mask')} with_mask, "
          f"{np.sum(labels == 'without_mask')} without_mask)")
    return data, labels


def build_model(lr, epochs):
    # Base model: MobileNetV2 pre-trained on ImageNet, without its top classifier.
    base_model = MobileNetV2(weights="imagenet", include_top=False,
                              input_tensor=Input(shape=(224, 224, 3)))

    # Custom classification head that we will train on top of the base model.
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    # Freeze the base layers so only the head is trained initially.
    for layer in base_model.layers:
        layer.trainable = False

    opt = Adam(learning_rate=lr, weight_decay=lr / epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)

    data, labels = load_dataset(args.dataset)

    # One-hot encode labels: with_mask / without_mask -> [1,0] / [0,1]
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    (train_x, test_x, train_y, test_y) = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )

    # Data augmentation to reduce overfitting on a small dataset.
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    model = build_model(args.lr, args.epochs)
    print(model.summary())

    print("[INFO] training head...")
    history = model.fit(
        aug.flow(train_x, train_y, batch_size=args.batch_size),
        steps_per_epoch=len(train_x) // args.batch_size,
        validation_data=(test_x, test_y),
        validation_steps=len(test_x) // args.batch_size,
        epochs=args.epochs,
    )

    print("[INFO] evaluating network...")
    pred_idxs = model.predict(test_x, batch_size=args.batch_size)
    pred_idxs = np.argmax(pred_idxs, axis=1)
    print(classification_report(
        test_y.argmax(axis=1), pred_idxs, target_names=lb.classes_
    ))

    print(f"[INFO] saving mask detector model to '{args.model}'...")
    model.save(args.model)

    # Plot training loss/accuracy.
    n = args.epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args.plot)
    print(f"[INFO] training plot saved to '{args.plot}'")


if __name__ == "__main__":
    main()