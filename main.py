import os
import json  # Make sure to import the json module
from ArabicTextDataset import ArabicTextDataset
from ArabicTextDataLoader import ArabicTextDataLoader
from ArabicTextClassifier import ArabicTextClassifier
from transformers import AutoModelForSequenceClassification as ArabertModel
from transformers import AutoTokenizer as ArabertTokenizer
import torch
import logging

def main():
    print("Current Working Directory:", os.getcwd())
    print("Directory exists:", os.path.exists('./model_checkpoints/'))

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)



    root_folder = os.path.abspath('Dataset')
    # root_folder = config['root_folder']  # or another path where the dataset is expected to be

    # Validate paths for 'Human' and 'AI' subdirectories under 'Training', 'Testing', and 'Validation'
    data_types = ['Training', 'Testing', 'Validation']
    for data_type in data_types:
        human_text_folder = os.path.join(root_folder, "Human", data_type)
        ai_text_folder = os.path.join(root_folder, "AI", data_type)

        for folder in [human_text_folder, ai_text_folder]:
            if not os.path.isdir(folder):
                logging.error(f"Error: {folder} does not exist or is not a directory.")
                return
#test

    model_name = config['model_name']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    batch_size = config['batch_size']



    model_path = config['final_model_path']  # or another path where the pretrained model is expected to be

    # Check if there is a saved model and load it, otherwise load the default model
    if os.path.isdir(model_path):
        logging.info(f"Loading model from {model_path}")
        model = ArabertModel.from_pretrained(model_path)
        tokenizer = ArabertTokenizer.from_pretrained(model_path)
    else:
        logging.info(f"No saved model found at {model_path}. Loading default model.")
        model = ArabertModel.from_pretrained(model_name)
        tokenizer = ArabertTokenizer.from_pretrained(model_name)

        # Constructing datasets - no changes here

    # Use the root_folder directly when constructing dataset instances
    train_dataset = ArabicTextDataset(root_folder, tokenizer, 'Training')
    val_dataset = ArabicTextDataset(root_folder, tokenizer, 'Validation')
    test_dataset = ArabicTextDataset(root_folder, tokenizer, 'Testing')

    # You should also use the 'batch_size' from your configuration here
    data_loader = ArabicTextDataLoader(train_dataset, val_dataset, test_dataset, batch_size=batch_size)  # changed from fixed number to variable
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()



    # Initialize the classifier
    classifier = ArabicTextClassifier(model=model, tokenizer=tokenizer, model_name=model_name, num_labels=2, epochs=epochs, learning_rate=learning_rate)

    # Train the model
    try:
        classifier.train(train_loader, val_loader)
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return  # Return early from main if training fails

        # Save the model after training
        save_path = os.path.join(model_path, 'AI_Arabic_Detector')
        classifier.save(save_path)

        # Evaluate the trained model on your test data
        classifier.evaluate(test_loader)

        # Plot metrics and continue with any further operations
        classifier.plot_metrics()



if __name__ == '__main__':
    main()
