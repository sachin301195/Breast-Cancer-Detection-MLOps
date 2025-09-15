import tensorflow as tf
import keras
import mlflow
import mlflow.tensorflow as mlt
import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
import collections

METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 16 # Keeping batch size small for CPU

def build_model(num_classes):
    """Builds the ResNetV2-based transfer learning model."""
    print("\n--- Building Model ---")
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False  # Starting with frozen base model

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)  # Increased dropout for regularization
    outputs = keras.layers.Dense(num_classes, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    
    print("Model built successfully.")
    model.summary()
    return model

def main(args):
    """Main training and evaluation loop."""
    mlflow.autolog(log_models=False, disable_for_unsupported_versions=True)
    
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(vars(args))

        # --- Data Augmentation and Generators ---
        print("\n--- Setting up Data Generators with Augmentation ---")
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation/Test data should not be augmented
        val_test_datagen = ImageDataGenerator()

        train_dir = 'data/processed/train'
        val_dir = 'data/processed/val'
        test_dir = 'data/processed/test'

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        validation_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )

        # --- Class Weight Calculation ---
        print("\n--- Calculating Class Weights ---")
        counter = collections.Counter(train_generator.classes)
        max_val = float(max(counter.values()))
        class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
        print(f"Class Weights: {class_weights}")
        mlflow.log_param("class_weights", class_weights)

        # --- Model Compilation ---
        model = build_model(num_classes=1)
        loss_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
        optimizer = AdamW(learning_rate=args.lr, weight_decay=1e-5)
        
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=METRICS)

        # --- Callbacks for Smarter Training ---
        # Stop training early if validation loss doesn't improve.
        # Reduce learning rate automatically when learning plateaus.
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)
        ]

        # --- STAGE 1: Transfer Learning (Train the Head) ---
        print("\n--- STAGE 1: Training the model head ---")
        history = model.fit(
            train_generator,
            epochs=args.epochs,
            validation_data=validation_generator,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # --- STAGE 2: Fine-Tuning (Train top layers of base model) ---
        print("\n--- STAGE 2: Fine-tuning the top layers ---")
        base_model = model.layers[1] # Correctly reference the base model
        base_model.trainable = True
        
        # Freeze all but the top 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Re-compile the model with a much lower learning rate for fine-tuning
        optimizer_ft = AdamW(learning_rate=args.lr / 10, weight_decay=1e-5)
        model.compile(optimizer=optimizer_ft, loss=loss_fn, metrics=METRICS)
        
        # Continue training
        history_ft = model.fit(
            train_generator,
            epochs=args.epochs,
            validation_data=validation_generator,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # --- Final Evaluation ---
        print("\n--- Evaluating Model on Test Set ---")
        results = model.evaluate(test_generator, return_dict=True)
        print(f"Test Results: {results}")

        for metric, value in results.items():
            mlflow.log_metric(f"test_{metric}", value)

        os.makedirs('models', exist_ok=True)
        model.save('models/breast_cancer_classifier_cpu_final.keras')
        print("\nModel saved to models/breast_cancer_classifier_cpu_final.keras")
        print(f"MLflow run completed. Run ID: {run.info.run_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs per stage')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial Learning rate')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Path to processed data directory')
    args = parser.parse_args()
    main(args)
