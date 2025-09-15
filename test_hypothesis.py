import tensorflow as tf
import keras
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DATA_DIR = 'data/processed/test'
MODEL_PATH = 'models/breast_cancer_classifier_cpu_final.keras'

def main():
    """
    Tests the 'double preprocessing' hypothesis.
    It loads the existing model and evaluates it using a corrected data pipeline
    that does NOT apply the redundant preprocessing step in the data generator.
    """
    print("--- Testing 'Double Preprocessing' Hypothesis ---")
    print("The goal is to see if the existing model performs well when fed raw, unprocessed data.")

    # Load the existing trained model
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Set up a test data generator WITHOUT the preprocessing_function.
    print("Setting up corrected test data generator (NO preprocessing)...")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator() # No preprocessing_function
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False 
    )

    # Evaluate the model with the corrected data pipeline
    print("\n--- Evaluating model with the corrected data pipeline ---")
    if test_generator.samples == 0:
        print("No images found in the test directory.")
        return

    results = model.evaluate(test_generator, return_dict=True)

    print("\n--- Hypothesis Test Results ---")
    print(f"  Loss:      {results['loss']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  AUC:       {results['auc']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print("---------------------------------")
    
    if results['auc'] > 0.95 and results['precision'] > 0.95:
        print("\n HYPOTHESIS CONFIRMED! The model works perfectly with the correct data pipeline.")
        print("The 'double preprocessing' was the issue.")
    else:
        print("\n Hypothesis was incorrect. The issue lies elsewhere.")

if __name__ == '__main__':
    main()
