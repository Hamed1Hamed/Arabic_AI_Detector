import os
import json
from ArabicTextDataset import ArabicTextDataset
from ArabicTextDataLoader import ArabicTextDataLoader
from ArabicTextClassifier import ArabicTextClassifier
from transformers import AutoModelForSequenceClassification as ArabertModel, AutoTokenizer, \
    AutoModelForSequenceClassification
from transformers import AutoTokenizer as ArabertTokenizer
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
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    batch_size = config['batch_size']
    model_path = config['final_model_path']

    # Check if there is a saved model and load it, otherwise load the default model
    if os.path.isdir(model_path):
        logging.info(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        logging.info(f"No saved model found at {model_path}. Loading default model.")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Constructing datasets using the data_type for clarity
    train_dataset = ArabicTextDataset(tokenizer, 'train')
    val_dataset = ArabicTextDataset(tokenizer, 'val')
    test_dataset = ArabicTextDataset(tokenizer, 'test')

    # Creating data loaders
    data_loader = ArabicTextDataLoader(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Initialize the classifier
    classifier = ArabicTextClassifier(model=model, tokenizer=tokenizer, model_name=model_name, num_labels=2, epochs=epochs, learning_rate=learning_rate)

    # Train the model
    try:
        classifier.train(train_loader, val_loader)
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return

    # Save the model after training
    save_path = os.path.join(model_path, 'Saved_AI_Arabic_Detector_Model')
    classifier.save(save_path)

    # Evaluate the trained model on the test data
    classifier.evaluate(test_loader)

    # Plot metrics
    classifier.plot_metrics()

if __name__ == '__main__':
    main()
