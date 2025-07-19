import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import re
from tqdm import tqdm

def create_metadata_file(data_dir, output_csv):
    """
    Extract:
    Walks through the data directory and creates a structured metadata into single CSV file. 
    Creating single source of truth for the data.
    """

    print("Phase E: Extracting metadata from the filenames...")
    records = []

    pattern = re.compile(r'SOB_([A-Z])_([A-Z]+)-(\d+-\d+)-(\d+)-\d+\.png')

    for root, _, files in os.walk(data_dir):  
        for file in files:
            if file.endswith('.png'):
                match = pattern.match(file)
                if match:
                    group, case_type, slide_id, magnification = match.groups()
                    patient_id = f"{case_type}-{slide_id}"
                    image_id = f"{group}-{magnification}"

                    records.append({
                        'file_name': file,
                        'file_path': os.path.join(root, file),
                        'patient_id': patient_id,
                        'image_id': image_id,
                        'magnification': magnification,
                        'class': 'benign' if group == 'B' else 'malignant', 
                        'case_type': case_type
                    })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Metadata for {len(df)} images created successfully: {output_csv}")

    return df

def split_data_by_patient(metadata_df, output_dir):
    """
    Transform:
    Splits the images into train, validation, and test sets based on the patient_id.
    """

    print("Phase T: Splitting data by patient...")
    df_filtered_400X = metadata_df[metadata_df['magnification'] == '400'].copy()
    print(f"Filtered data with 400X magnification, resulting in {len(df_filtered_400X)} images out of total {len(metadata_df)} images.")

    patient_ids = df_filtered_400X['patient_id'].unique()

    # Spliting patient IDs (70% train, 15% val, 15% test)
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=21)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=21)

    for split in ['train', 'val', 'test']:
        for class_name in ['benign', 'malignant']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    def assign_split(patient_id):
        if patient_id in train_ids:
            return 'train'
        elif patient_id in val_ids:
            return 'val'
        else:
            return 'test'
    
    df_filtered_400X['split'] = df_filtered_400X['patient_id'].apply(assign_split)

    for _, row in tqdm(df_filtered_400X.iterrows(), total=df_filtered_400X.shape[0], desc="Copying images"):
        dest_dir = os.path.join(output_dir, row['split'], row['class'])
        shutil.copy(row['file_path'], dest_dir)
    
    print("Data splitting and copying complete.")
    df_filtered_400X.to_csv(os.path.join(output_dir, 'metadata_with_splits.csv'), index=False)



if __name__ == "__main__":
    RAW_DATA_DIR = 'data/raw/BreaKHis_v1/histology_slides/breast'
    PROCESSED_DATA_DIR = 'data/processed'
    METADATA_CSV = 'data/metadata.csv'
    print(f"Starting ETL process...")

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory {RAW_DATA_DIR} does not exist.")
    else:
        metadata_df = create_metadata_file(RAW_DATA_DIR, METADATA_CSV)
        split_data_by_patient(metadata_df, PROCESSED_DATA_DIR)

        print("\n ETL Process finished successfully!")
        print(f"Processed data is ready in {PROCESSED_DATA_DIR}")
