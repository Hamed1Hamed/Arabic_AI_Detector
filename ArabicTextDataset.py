import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging
import re
import json
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
class ArabicTextDataset(Dataset):
    def __init__(self, root_folder, tokenizer, data_type):
        """
        :param root_folder: The root folder containing the 'Dataset' folder, which then contains 'human' and 'ai' folders.
        :param tokenizer: The tokenizer instance.
        :param data_type: Type of data - 'train', 'val', or 'test'.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Initializing ArabicTextDataset')
        # At the beginning of your script, load the configuration

        # Validate the root folder
        if not os.path.exists(root_folder):
            logging.error(f"Directory not found: {root_folder}")
            raise FileNotFoundError(f"{root_folder} does not exist or is not accessible.")

        self.tokenizer = tokenizer  # We directly accept the tokenizer instance

        # Construct the path to the 'Dataset' directory
        dataset_folder = root_folder


        # Define the folders based on the type of data
        self.human_text_folder = os.path.join(dataset_folder, "Human", data_type)  # e.g., "Dataset/Human/train"
        self.ai_text_folder = os.path.join(dataset_folder, "AI", data_type)  # e.g., "Dataset/AI/train"

        # Validate the subfolders
        for folder in [self.human_text_folder, self.ai_text_folder]:
            if not os.path.exists(folder):
                logging.error(f"Directory not found: {folder}")
                raise FileNotFoundError(f"{folder} does not exist or is not accessible.")

        self.samples = []

        # Load the dataset
        self._load_folder(self.human_text_folder, label=0)  # human texts with label 0
        self._load_folder(self.ai_text_folder, label=1)  # AI-generated texts with label 1

        if len(self.samples) == 0:
            logging.error("No valid data samples found.")
            raise RuntimeError("No valid data samples found.")

        logging.info(f"Loaded {len(self.samples)} samples.")

    def _load_folder(self, folder_path, label):
        """
        Load all text files from the specified folder with the given label.
        """
        for fname in os.listdir(folder_path):
            if fname.endswith('.txt'):
                file_path = os.path.join(folder_path, fname)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    # Apply preprocessing to the text
                    text = preprocess_arabic(text)

                    if text:
                        self.samples.append((text, label))
                    else:
                        logging.warning(f"Empty file: {file_path}")

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

