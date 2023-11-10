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

        """
        :param tokenizer: The tokenizer instance.
        :param data_type: Type of data - 'train', 'val', or 'test'.
        """
        self.indicator_phrases = load_indicator_phrases(indicator_phrases_path)
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
            if df.empty:
                logging.error("CSV file is empty: %s", file_path)
                raise ValueError("CSV file is empty")

            for index, row in df.iterrows():
                text = row['text']
                #char_count = row['Char_Count']
                label = int(row['label'])
                #self.samples.append((text, char_count, label))
                self.samples.append((text, label))
        except pd.errors.EmptyDataError:
            logging.error("CSV file is empty: %s", file_path)
            raise
        except pd.errors.ParserError as e:
            logging.error("Error parsing the CSV file at %s: %s", file_path, e)
            raise
        except FileNotFoundError as e:
            logging.error("CSV file not found: %s", file_path)
            raise
        except Exception as e:  # You can handle any other unforeseen exceptions here.
            logging.error("An unexpected error occurred while reading the CSV file at %s: %s", file_path, e)
            raise

    def _contains_indicator_phrases(self, text):
        return any(phrase in text for phrase in self.indicator_phrases)
    def __len__(self):
        return len(self.samples)

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

            """
            :param tokenizer: The tokenizer instance.
            :param data_type: Type of data - 'train', 'val', or 'test'.
            """
            self.indicator_phrases = load_indicator_phrases(indicator_phrases_path)
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
                if df.empty:
                    logging.error("CSV file is empty: %s", file_path)
                    raise ValueError("CSV file is empty")

                for index, row in df.iterrows():
                    text = row['text']
                    # char_count = row['Char_Count']
                    label = int(row['label'])
                    # self.samples.append((text, char_count, label))
                    self.samples.append((text, label))
            except pd.errors.EmptyDataError:
                logging.error("CSV file is empty: %s", file_path)
                raise
            except pd.errors.ParserError as e:
                logging.error("Error parsing the CSV file at %s: %s", file_path, e)
                raise
            except FileNotFoundError as e:
                logging.error("CSV file not found: %s", file_path)
                raise
            except Exception as e:  # You can handle any other unforeseen exceptions here.
                logging.error("An unexpected error occurred while reading the CSV file at %s: %s", file_path, e)
                raise

        def _contains_indicator_phrases(self, text):
            return any(phrase in text for phrase in self.indicator_phrases)

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
            # Check if the text contains any of the AI indicator phrases
            ai_indicator = torch.tensor([self._contains_indicator_phrases(text)], dtype=torch.float)

            # Return the encoding, the AI indicator, and the label
            return {key: val.squeeze(0) for key, val in encoding.items()}, ai_indicator, torch.tensor(label,
                                                                                                      dtype=torch.long)

