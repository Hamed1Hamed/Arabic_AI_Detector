import os
import json
from ArabicTextDataset import ArabicTextDataset
from ArabicTextDataLoader import ArabicTextDataLoader
from ArabicTextClassifier import ArabicTextClassifier
from transformers import AutoTokenizer
import logging
"""
 the best model will be saved in the "model_checkpoints" folder with the filename "best_model.pt".
 after training and evaluation, your model will be saved in a folder named "final_model" with the filename "Saved_AI_Arabic_Detector_Model"
 
"""
def main():
    print("Current Working Directory:", os.getcwd())
    print("Directory exists:", os.path.exists('./model_checkpoints/'))
    # Check if the 'final_model' directory exists

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    model_name = config['model_name']
    batch_size = config['batch_size']
    model_path = config['final_model_path']
    indicator_phrases_path = config['indicator_phrases_path']  # Load the indicator phrases path
    model_checkpoints= config['checkpoint_path'] # Load the model checkpoints path

    final_model_path = config['final_model_path']  # Assuming 'config' is already loaded from 'config.json'
    print("Directory exists ('final_model'):", os.path.exists(f'./{final_model_path}/'))

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
        checkpoint_path=model_checkpoints
    )

    # Load the best model if it exists, else start training from scratch
    best_model_path = os.path.join(model_checkpoints, "best_model.pt")
    if os.path.isfile(best_model_path):
        print("Loading the best model...")
        classifier.load_best_model()
    else:
        print("No best model found. Starting training from scratch.")

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
        save_path = os.path.join(model_path, r'Saved_AI_Arabic_Detector_Model')
        print(f"Attempting to save model to: {save_path}")
        classifier.save(save_path)

        # Check if the best model was saved during training
        if os.path.isfile(best_model_path):
            print("The best model was saved successfully.")
        else:
            print("The best model was not saved.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
