import json
import logging
from torch.utils.data import DataLoader, Dataset

class ArabicTextDataLoader:
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size):

        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Checking the type of the provided datasets
        if not isinstance(train_dataset, Dataset) or not isinstance(val_dataset, Dataset) or not isinstance(test_dataset, Dataset):
            logging.error("Invalid dataset type provided.")  # Logging the error before raising the exception
            raise TypeError("Provided dataset is not a valid PyTorch Dataset.")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def get_data_loaders(self):
        # No need to split the dataset, just create DataLoaders directly

        # Creating data loaders and logging the process
        logging.info("Creating data loaders with appropriate batch size and shuffle parameters")

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)  # test loader added

        return train_loader, val_loader, test_loader  # return all three loaders

