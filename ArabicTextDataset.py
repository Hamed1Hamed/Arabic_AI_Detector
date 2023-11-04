import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging
import re
import json
import pandas as pd



# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)
dataset_folder = config['root_folder']


def preprocess_arabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("[ﻷﻵﻹﻻ]", "لا", text)
    #text = re.sub(r'[^ا-ي0-9\s]', '', text)  # Remove all non-Arabic-characters and non-digits
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text.strip()


# Define the mapping from data_type to CSV filenames
data_type_to_filename = {
    'train': 'Training.xlsx',
    'val': 'Validation.xlsx',
    'test': 'Testing.xlsx'
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
        # ...
        self._load_excel(file_path)


        if len(self.samples) == 0:
            logging.error("No valid data samples found.")
            raise RuntimeError("No valid data samples found.")

        logging.info(f"Loaded {len(self.samples)} samples.")

    def _load_excel(self, file_path):
        try:
            df = pd.read_excel(file_path)
            # Assuming that the dataframe has two columns: 'text' for the review and 'label' for the sentiment
            for index, row in df.iterrows():
                text = row['text']  # Directly use the text without preprocessing
                label = int(row['label'])  # Replace 'label' with the actual column name for your labels
                self.samples.append((text, label))
        except Exception as e:
            raise IOError(f"Error reading the Excel file: {e}")

    # def _load_excel(self, file_path):
    #     try:
    #         df = pd.read_excel(file_path)
    #         # Assuming that the dataframe has two columns: 'text' for the review and 'label' for the sentiment
    #         for index, row in df.iterrows():
    #             text = preprocess_arabic(row['text'])  # Replace 'text' with the actual column name for your text data
    #             label = int(row['label'])  # Replace 'label' with the actual column name for your labels
    #             self.samples.append((text, label))
    #     except Exception as e:
    #         raise IOError(f"Error reading the Excel file: {e}")

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
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(label, dtype=torch.long)
