import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
import json

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)
dataset_folder = config['root_folder']

# Define the mapping from data_type to CSV filenames
data_type_to_filename = {
    'train': 'Training.csv',
    'val': 'Validation.csv',
    'test': 'Testing.csv'
}

class ArabicTextDataset(Dataset):
    def __init__(self, tokenizer, data_type):
        """
        :param tokenizer: The tokenizer instance.
        :param data_type: Type of data - 'train', 'val', or 'test'.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Initializing ArabicTextDataset')

        # Validate the dataset folder
        if not os.path.isdir(dataset_folder):
            logging.error(f"Directory not found: {dataset_folder}")
            raise FileNotFoundError(f"{dataset_folder} does not exist or is not accessible.")

        self.tokenizer = tokenizer

        # Get the correct file name from the mapping
        file_name = data_type_to_filename.get(data_type.lower())
        if not file_name:
            raise ValueError(f"Invalid data_type provided: {data_type}. Expected 'train', 'val', or 'test'.")

        file_path = os.path.join(dataset_folder, file_name)

        # Validate the CSV file path
        if not os.path.isfile(file_path):
            logging.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"{file_path} does not exist or is not accessible.")

        self.samples = []

        # Load the dataset from the CSV file
        self._load_csv(file_path)

        if len(self.samples) == 0:
            logging.error("No valid data samples found.")
            raise RuntimeError("No valid data samples found.")

        logging.info(f"Loaded {len(self.samples)} samples.")

    def _load_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                text = row['text']  # Directly use the text without preprocessing
                char_count = row['Char_Count']  # The new Char_Count column
                label = int(row['label'])  # Convert label to integer
                self.samples.append((text, char_count, label))
        except Exception as e:
            raise IOError(f"Error reading the CSV file: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, char_count, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )
        # You can now return Char_Count if you plan to use it as a feature
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(char_count), torch.tensor(label, dtype=torch.long)
