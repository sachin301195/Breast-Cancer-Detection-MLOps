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
    print("Building model...")
    base_model=keras.applications.ResNet50V2(
        input_shape=(224, 224, 3), 
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable=False

    inputs=keras.Input(shape=(224, 224, 3))
    x=keras.applications.resnet_v2.preprocess_input(inputs)
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

def main(epochs, learning_rate):
    mlt.autolog()

    train_ds, val_ds, test_ds = load_data('data/processed')
    model=build_model(learning_rate)

    print("\n--- Starting Model Training ---")
    with mlflow.start_run() as run:
        model.fit(
            train_ds, 
            epochs=epochs, 
            validation_data=val_ds
        )

        print("\n--- Evaluating Model on Test Set ---")
        loss, accuracy = model.evaluate(test_ds)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        os.makedirs('models', exist_ok=True)
        model.save('models/breast_cancer_classifier.keras')
        print("\nModel saved to models/breast_cancer_classifier.keras")
        print(f"MLflow run completed. Run ID: {run.info.run_id}")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args=parser.parse_args()

    main(epochs=args.epochs, learning_rate=args.lr)
