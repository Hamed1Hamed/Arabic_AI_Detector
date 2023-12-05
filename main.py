import os
import json
from ArabicTextDataset import ArabicTextDataset
from ArabicTextDataLoader import ArabicTextDataLoader
from ArabicTextClassifier import ArabicTextClassifier
from transformers import AutoTokenizer
import logging
import random
import numpy as np
import torch



def main():

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    # Set seed for reproducibility
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If you are using CUDA (PyTorch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Current Working Directory: {}".format(os.getcwd()))
    logger.info("Directory exists: {}".format(os.path.exists('./model_checkpoints/')))

    # Set up full paths for saving models and checkpoints
    project_root_dir = os.getcwd()  # Assumes the project root directory is the current working directory
    model_save_path = os.path.join(project_root_dir, config['model_save_path'])
    final_model_path = os.path.join(project_root_dir, config['final_model_path'])
    model_checkpoints = os.path.join(project_root_dir, config['checkpoint_path'])
    batch_size = config['batch_size']

    # Ensure that directories exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(final_model_path, exist_ok=True)
    os.makedirs(model_checkpoints, exist_ok=True)

    # Check if the directories exist (optional, for verification)
    logger.info(f"Directory exists: {os.path.exists(model_save_path)}")
    logger.info(f"Directory exists: {os.path.exists(final_model_path)}")
    logger.info(f"Directory exists: {os.path.exists(model_checkpoints)}")

    # Load tokenizer

    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize datasets
    train_dataset = ArabicTextDataset(tokenizer, 'train')
    val_dataset = ArabicTextDataset(tokenizer, 'val')
    test_dataset = ArabicTextDataset(tokenizer, 'test')

    # Initialize data loaders
    data_loader = ArabicTextDataLoader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        seed=config['seed'],
        num_workers=4,
        pin_memory=True
    )
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Initialize the classifier with the new parameters
    classifier = ArabicTextClassifier(
        model_name=config['model_name'],
        num_labels=2,
        learning_rate=config['learning_rate'],
        epochs=config['epochs'],
        checkpoint_path=model_checkpoints,
        patience=config['patience'],
        initial_learning_rate=config['initial_learning_rate'],
        warmup_epochs=config['warmup_epochs']
    )

    # Load the best model if it exists, else start training from scratch
    best_model_path = os.path.join(model_checkpoints, "best_model.pt")
    if os.path.isfile(best_model_path):
        logger.info("Loading the best model...")
        classifier.load_best_model()
    else:
        logger.info("No best model found. Starting training from scratch.")

    # Move classifier to the appropriate device
    classifier.to(classifier.device)

    # Train and evaluate
    try:
        final_train_accuracy, final_test_accuracy = classifier.train(train_loader, val_loader, test_loader)

        # Plot training/validation metrics
        classifier.plot_metrics()

        # Plot final accuracies
        classifier.plot_final_accuracies(final_train_accuracy, final_test_accuracy)

        # Save final model
        final_model_save_path = os.path.join(final_model_path, "Saved_AI_Arabic_Detector_Model.pt")
        logger.info(f"Attempting to save the final model to: {final_model_save_path}")
        classifier.save(final_model_save_path)

        # Check if the best model was saved during training
        if os.path.isfile(best_model_path):
            logger.info("The best model was saved successfully.")
        else:
            logger.info("The best model was not saved.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()
