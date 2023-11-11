import os
import json
from ArabicTextDataset import ArabicTextDataset
from ArabicTextDataLoader import ArabicTextDataLoader
from ArabicTextClassifier import ArabicTextClassifier
from transformers import AutoTokenizer
import logging

def main():
    print("Current Working Directory:", os.getcwd())
    print("Directory exists:", os.path.exists('./model_checkpoints/'))

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    model_name = config['model_name']
    batch_size = config['batch_size']
    model_path = config['final_model_path']
    indicator_phrases_path = config['indicator_phrases_path']  # Load the indicator phrases path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Initialize datasets
    train_dataset = ArabicTextDataset(tokenizer, 'train', indicator_phrases_path)
    val_dataset = ArabicTextDataset(tokenizer, 'val', indicator_phrases_path)
    test_dataset = ArabicTextDataset(tokenizer, 'test', indicator_phrases_path)

    # Initialize data loaders
    data_loader = ArabicTextDataLoader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Initialize the classifier
    classifier = ArabicTextClassifier(
        model_name=model_name,
        num_labels=2,
        learning_rate=config['learning_rate'],
        epochs=config['epochs'],
        checkpoint_path=config['checkpoint_path']
    )

    # Move classifier to the appropriate device
    classifier.to(classifier.device)

    # Train and evaluate
    try:
        classifier.train(train_loader, val_loader)

        # Evaluate on test data
        classifier.evaluate(test_loader)

        # Plot training/validation metrics
        classifier.plot_metrics()

        # Save final model
        save_path = os.path.join(config['final_model_path'], 'Saved_AI_Arabic_Detector_Model')
        classifier.save(save_path)

        # Check if the best model was saved during training
        checkpoint_dir = config['checkpoint_path']  # Use the path from the configuration
        if os.path.isfile(os.path.join(checkpoint_dir, "best_model.pt")):
            print("The best model was saved successfully.")
        else:
            print("The best model was not saved.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()