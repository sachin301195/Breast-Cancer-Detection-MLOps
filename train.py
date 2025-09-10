import tensorflow as tf
import keras
import mlflow
import mlflow.tensorflow as mlt
import os
import argparse
from typing import Tuple, cast


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_data(data_dir):
    print("Loading Data...")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_df=cast(tf.data.Dataset, keras.utils.image_dataset_from_directory(
        train_dir, label_mode='binary', image_size=IMG_SIZE, batch_size=BATCH_SIZE
    ))
    val_df=cast(tf.data.Dataset, keras.utils.image_dataset_from_directory(
        val_dir, label_mode='binary', image_size=IMG_SIZE, batch_size=BATCH_SIZE
    ))
    test_df=cast(tf.data.Dataset, keras.utils.image_dataset_from_directory(
        test_dir, label_mode='binary', image_size=IMG_SIZE, batch_size=BATCH_SIZE
    ))

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset=train_df.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset=val_df.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset=test_df.cache().prefetch(buffer_size=AUTOTUNE)

    print("Data loaded successfully.")
    return train_dataset, val_dataset, test_dataset

def build_model(learning_rate):
    print("Building model with Data Augmentation and Increased Dropout...")

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2),
    ])

    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = keras.applications.resnet_v2.preprocess_input(x)
    
    base_model=keras.applications.ResNet50V2(
        input_shape=(224, 224, 3), 
        include_top=False,
        weights='imagenet', 
        name='resnet50v2'
    )
    base_model.trainable=False

    x=base_model(x, training=False)
    x=keras.layers.GlobalAveragePooling2D()(x)
    x=keras.layers.Dropout(0.2)(x)
    outputs=keras.layers.Dense(1, activation='sigmoid')(x)

    model=keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=keras.losses.BinaryCrossentropy(), 
        metrics=['accuracy']
    )

    print('Model built successfully.')
    model.summary()

    return model

# The NEW code for train.py's main function

def main(epochs, learning_rate):
    mlt.autolog()

    train_ds, val_ds, test_ds = load_data('data/processed')
    model = build_model(learning_rate)

    # === Calculate Class Weights ===
    neg, pos = 126, 833 # Benign, Malignant
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Class Weights:\n  - Benign (0): {class_weight[0]:.2f}\n  - Malignant (1): {class_weight[1]:.2f}")

    # === Single, Stable Training Run ===
    print("\n--- Starting Stable Model Training with Class Weights---")
    
    # Use EarlyStopping to find the best model automatically
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5, # Allow 5 epochs for improvement before stopping
        restore_best_weights=True
    )

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[early_stopping_callback],
        class_weight=class_weight  # Apply the crucial class weights
    )

    # --- Evaluation ---
    print("\n--- Evaluating Final Model on Test Set ---")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)

    os.makedirs('models', exist_ok=True)
    model.save('models/breast_cancer_classifier.keras')
    print("\nModel saved to models/breast_cancer_classifier.keras")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args=parser.parse_args()

    main(epochs=args.epochs, learning_rate=args.lr)
