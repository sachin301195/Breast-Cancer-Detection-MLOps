import tensorflow as tf
import keras
import os
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryFocalCrossentropy

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DATA_DIR = 'data/processed/test'
MODEL_PATH = 'models/breast_cancer_classifier_cpu_final.keras'

def main():
    """
    Loads the final saved model WITH custom objects and evaluates it 
    on the test set to confirm its performance.
    """
    print("--- Model Validation Script (with Custom Objects) ---")

    # Define the custom objects the model was trained with
    custom_objects = {
        'AdamW': AdamW,
        'BinaryFocalCrossentropy': BinaryFocalCrossentropy
    }
    
    # Load the trained model, passing the custom objects
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Set up the test data generator
    print(f"Setting up test data generator from: {TEST_DATA_DIR}")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Evaluate the model
    print("\n--- Evaluating model on the test set ---")
    if test_generator.samples == 0:
        print("No images found in the test directory.")
        return

    results = model.evaluate(test_generator, return_dict=True)

    print("\n--- Validation Results ---")
    print(f"  Loss:      {results['loss']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  AUC:       {results['auc']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print("--------------------------")
    
    if results['auc'] > 0.95 and results['precision'] > 0.95:
        print("\n SUCCESS! The model's performance is excellent again.")
    else:
        print("\n Still seeing low performance. The issue is something else.")


if __name__ == '__main__':
    main()

