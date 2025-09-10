import tensorflow as tf
import keras
import mlflow
import mlflow.tensorflow as mlt
import os
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, cast
from imblearn.over_sampling import ADASYN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.model_selection import train_test_split

# --- Define Custom Metrics ---
METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 16 # Reduced batch size for CPU

# --- Updated Data Loading for CPU ---
def load_and_preprocess_data(data_dir, split, fraction=1.0, augment=False):
    """
    Loads data and creates a generator.
    For the training set, it can load a smaller, stratified fraction of the data.
    """
    data_path = os.path.join(data_dir, split)

    image_paths = []
    labels = []
    class_map = {'benign': 0, 'malignant': 1}
    for class_name, label in class_map.items():
        class_path = os.path.join(data_path, class_name)
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))
            labels.append(label)

    if split == 'train' and fraction < 1.0:
        print(f"Using {fraction*100:.0f}% of the training data for this run.")
        image_paths, _, labels, _ = train_test_split(
            image_paths, labels, train_size=fraction, stratify=labels, random_state=42
        )

    print(f"Loading {len(image_paths)} images into memory...")
    images = np.array([keras.preprocessing.image.load_img(p, target_size=IMG_SIZE) for p in image_paths])
    images = np.array([keras.preprocessing.image.img_to_array(img) for img in images])
    labels = np.array(labels)

    if split == 'train':
        print(f"Original training data shape: {images.shape}")
        reshaped_images = images.reshape(len(images), -1)
        adasyn = ADASYN(random_state=42)
        print("Starting ADASYN resampling... This should be much faster now.")
        X_resampled, y_resampled = adasyn.fit_resample(reshaped_images, labels)
        print("ADASYN complete.")
        images = X_resampled.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 3)
        labels = y_resampled
        print(f"Resampled training data shape with ADASYN: {images.shape}")

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            preprocessing_function=keras.applications.resnet_v2.preprocess_input
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=keras.applications.resnet_v2.preprocess_input
        )

    def generator():
        for batch_x, batch_y in datagen.flow(images, labels, batch_size=BATCH_SIZE, shuffle=True):
            yield batch_x, batch_y

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return dataset.prefetch(buffer_size=AUTOTUNE)

def build_model():
    print("Building model...")
    base_model = keras.applications.ResNet50V2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    print('Model built successfully.')
    return model

def main(args):
    mlt.autolog()

    print("Loading Data with new pipeline...")
    train_ds = load_and_preprocess_data('data/processed', 'train', fraction=args.data_fraction, augment=True)
    val_ds = load_and_preprocess_data('data/processed', 'val')
    test_ds = load_and_preprocess_data('data/processed', 'test')

    model = build_model()

    # --- THIS LINE IS NOW FIXED ---
    train_df = pd.read_csv('data/processed/metadata_with_splits.csv')
    train_files = len(train_df[train_df['split'] == 'train'])
    if args.data_fraction < 1.0:
        train_files = int(train_files * args.data_fraction)

    decay_steps = (train_files // BATCH_SIZE) * args.epochs

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr, decay_steps=decay_steps if decay_steps > 0 else 1
    )

    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    loss_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

    with mlflow.start_run() as run:
        # --- STAGE 1 ---
        print("\n--- Starting Training: STAGE 1 (Classifier Head) ---")
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=METRICS)
        model.fit(
            train_ds, epochs=args.epochs, validation_data=val_ds,
            steps_per_epoch=train_files // BATCH_SIZE if train_files > BATCH_SIZE else 1
        )

        # --- STAGE 2 ---
        print("\n--- Starting Training: STAGE 2 (Fine-tuning top layers) ---")
        base_model = model.layers[1]
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        optimizer_ft = AdamW(learning_rate=args.lr / 10, weight_decay=1e-4)
        model.compile(optimizer=optimizer_ft, loss=loss_fn, metrics=METRICS)
        model.fit(
            train_ds, epochs=args.epochs, validation_data=val_ds,
            steps_per_epoch=train_files // BATCH_SIZE if train_files > BATCH_SIZE else 1
        )

        print("\n--- Evaluating Model on Test Set ---")
        results = model.evaluate(test_ds, return_dict=True)
        print(f"Test Results: {results}")

        for metric, value in results.items():
            mlflow.log_metric(f"test_{metric}", value)

        os.makedirs('models', exist_ok=True)
        model.save('models/breast_cancer_classifier_cpu_test.keras')
        print("\nModel saved to models/breast_cancer_classifier_cpu_test.keras")
        print(f"MLflow run completed. Run ID: {run.info.run_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs per stage')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial Learning rate')
    parser.add_argument(
        '--data-fraction',
        type=float,
        default=0.1,
        help='Fraction of the training data to use for a quick CPU run'
    )
    args = parser.parse_args()

    main(args)