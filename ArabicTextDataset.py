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


def load_indicator_phrases(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


class ArabicTextDataset(Dataset):
    def __init__(self, tokenizer, data_type, indicator_phrases_path):
        self.indicator_phrases = load_indicator_phrases(indicator_phrases_path)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Initializing ArabicTextDataset')

        # Validate the dataset folder
        if not os.path.isdir(dataset_folder):
            logging.error(f"Directory not found: {dataset_folder}")
            raise FileNotFoundError(f"{dataset_folder} does not exist or is not accessible.")

        self.tokenizer = tokenizer
        file_name = data_type_to_filename.get(data_type.lower())
        if not file_name:
            raise ValueError(f"Invalid data_type provided: {data_type}. Expected 'train', 'val', or 'test'.")

        file_path = os.path.join(dataset_folder, file_name)
        if not os.path.isfile(file_path):
            logging.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"{file_path} does not exist or is not accessible.")

        self.samples = []
        self._load_csv(file_path)

        if len(self.samples) == 0:
            logging.error("No valid data samples found.")
            raise RuntimeError("No valid data samples found.")

        logging.info(f"Loaded {len(self.samples)} samples.")

    def _load_csv(self, file_path):
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            text = row['text']
            label = int(row['label'])
            char_count = int(row['Char_Count'])  # Load Char_Count feature
            self.samples.append((text, char_count, label))  # Include Char_Count in samples

    def _contains_indicator_phrases(self, text):
        return any(phrase in text for phrase in self.indicator_phrases)

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
        ai_indicator = torch.tensor([self._contains_indicator_phrases(text)], dtype=torch.float)
        char_count_feature = torch.tensor([char_count], dtype=torch.float)  # Create Char_Count feature tensor

        # Include Char_Count feature in the input dictionary
        input_dict = {key: val.squeeze(0) for key, val in encoding.items()}
        input_dict['char_count'] = char_count_feature

        return input_dict, ai_indicator, torch.tensor(label, dtype=torch.long)
