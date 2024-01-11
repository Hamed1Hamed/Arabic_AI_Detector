import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
import json
from camel_tools.utils.dediac import dediac_ar

with open('config.json') as config_file:
    config = json.load(config_file)
dataset_folder = config['root_folder']

data_type_to_filename = {
    'train': 'Training.csv',
    'val': 'Validation.csv',
    'test': 'Testing.csv'
}

class ArabicTextDataset(Dataset):
    def __init__(self, tokenizer, data_type):
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
        self._load_csv(file_path, data_type)

        if len(self.samples) == 0:
            logging.error("No valid data samples found.")
            raise RuntimeError("No valid data samples found.")

        logging.info(f"Loaded {len(self.samples)} samples.")

# Apply dediacritization layer for evaluating the AIRABIC Dataset purpose. If you are using another dataset, you can remove this layer by the following
    # remove data_type argument from the _load_csv function. Then remove the if statment below that contains the dediacritization layer.
    def _load_csv(self, file_path, data_type):
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            text = row['text']
            label = int(row['label'])
            # Apply dediacritization layer for evaluating the AIRABIC Dataset purpose only. If you are using another dataset, you can remove this layer.
            if data_type == 'test':
                text = dediac_ar(text)  # Remove diacritics for testing set

            self.samples.append((text, label))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )


        input_dict = {key: val.squeeze(0) for key, val in encoding.items()}

        return input_dict, torch.tensor(label, dtype=torch.long)
