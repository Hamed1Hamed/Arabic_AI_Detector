from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import logging
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
logging.basicConfig(filename='classifier.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
import json
class ArabicTextClassifier:
   # classifier = ArabicTextClassifier(model=model, tokenizer=tokenizer, model_name=model_name, num_labels=2, epochs=epochs, learning_rate=learning_rate)

    def __init__(self, model, tokenizer, model_name, num_labels, epochs, learning_rate):
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initializing the ArabicTextClassifier.')
        self.checkpoint_path = config["checkpoint_path"]  # ensure this config key exists
        # self.patience = patience
        # Model setup
        if model is not None:
            self.model = model
        elif model_name is not None:
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            except Exception as e:
                error_message = f"Failed to load model. Error: {str(e)}"
                self.logger.error(error_message)
                raise ValueError(error_message)
        else:
            raise ValueError("Either 'model' or 'model_name' must be provided.")

        # Tokenizer setup
        if tokenizer is not None:
            self.tokenizer = tokenizer  # Use the provided tokenizer instance
        else:
            # Load the tokenizer if not provided
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                error_message = f"Failed to load tokenizer. Error: {str(e)}"
                self.logger.error(error_message)  # Logging the error before raising
                raise ValueError(error_message)

        # Set the other attributes
        self.num_labels = num_labels
        self.epochs = epochs
        self.learning_rate = learning_rate



        # Check device availability (CUDA)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device in use: {self.device}")
        self.model.to(self.device)

        # Attempt to setup the optimizer with exception handling
        try:
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
            self.logger.info("Optimizer initialized successfully.")
        except Exception as e:
            error_message = f"Failed to initialize optimizer. Error: {str(e)}"
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Initialize the lists for tracking progress
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # These are used during independent evaluation
        self.evaluation_losses = []
        self.evaluation_accuracies = []




    #----------------------------------------------------------------Training and Evaluation Functions-------------------------------------------------------------


    def train(self, train_loader, val_loader, start_epoch=0):
        best_val_loss = float('inf')
        epochs_without_improvement = 0  # Counter for early stopping

        if not isinstance(train_loader, DataLoader) or not isinstance(val_loader, DataLoader):
            raise TypeError("train_loader and val_loader must be DataLoader instances.")

        # Initialize the scheduler
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.logger.info('Training process started.')

        for epoch in range(start_epoch, self.epochs):
            self.model.train()  # Begin training
            total_train_loss = 0
            correct_train_preds = 0
            total_train_preds = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)

            for batch in progress_bar:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                inputs['labels'] = labels  # Add 'labels' to the inputs dictionary for loss computation

                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = outputs.loss

                # Validate loss is a valid scalar tensor
                if loss is None or not torch.is_tensor(loss) or loss.numel() != 1:
                    self.logger.error("Loss is not a valid scalar tensor.")
                    continue  # Skip this batch

                loss_value = loss.item()
                total_train_loss += loss_value
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct_train_preds += torch.sum(predictions == labels).item()
                total_train_preds += labels.size(0)

                # Calculate and display the running training accuracy
                train_accuracy = correct_train_preds / total_train_preds if total_train_preds > 0 else 0.0
                progress_bar.set_postfix(loss=loss_value, accuracy=train_accuracy)

            # Calculate average train loss and accuracy
            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = correct_train_preds / max(total_train_preds, 1)
            self.logger.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Train Acc: {train_accuracy}")

            # Validation step
            avg_val_loss, val_accuracy = self.evaluate(val_loader)

            # Scheduler step with the validation loss
            scheduler.step(avg_val_loss)

            # Checkpoint if this is the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                self.save_best_model()  # Save only the best model
            else:
                epochs_without_improvement += 1

            # Early stopping (if uncommented)
            # if epochs_without_improvement >= self.patience:
            #     self.logger.info("Early stopping triggered.")
            #     break

            # Record metrics for each epoch
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)

        self.logger.info('Training process has ended.')




    def evaluate(self, val_loader):
            self.model.eval()
            y_true = []
            y_pred = []
            correct_preds = 0
            total_preds = 0

            # Storage for evaluation metrics if they don't already exist
            if not hasattr(self, 'evaluation_accuracies'):
                self.evaluation_accuracies = []

            progress_bar = tqdm(val_loader, desc="Evaluating", leave=True)

            with torch.no_grad():
                for batch in progress_bar:
                    inputs, labels = batch
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    labels = labels.to(self.device)

                    outputs = self.model(**inputs)  # Forward pass

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predictions.cpu().numpy())

                    correct_preds += (predictions == labels).sum().item()
                    total_preds += labels.size(0)

            val_accuracy = correct_preds / total_preds
            # Log or return these metrics as needed
            self.logger.info(f"Validation Accuracy: {val_accuracy}")
            self.logger.info(f"Precision: {precision_score(y_true, y_pred, zero_division=0)}")
            self.logger.info(f"Recall: {recall_score(y_true, y_pred, zero_division=0)}")
            self.logger.info(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0)}")
            self.logger.info(f"AUC-ROC: {roc_auc_score(y_true, y_pred)}")

            # Append to the evaluation metrics lists
            self.evaluation_accuracies.append(val_accuracy)

            self.plot_confusion_matrix(y_true, y_pred)

#save()	Saves the model and tokenizer at a specified file path.
#save_best_model()	Saves the best model and tokenizer based on a specific metric.
    def save_best_model(self):
        # Save the best model without including the epoch in the filename
        model_save_path = os.path.join(self.checkpoint_path, "best_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        # Save the tokenizer in the same way (without epoch in the name if desired)
        tokenizer_save_path = os.path.join(self.checkpoint_path, "best_tokenizer")
        self.tokenizer.save_pretrained(tokenizer_save_path)
        self.logger.info(f"Best model and tokenizer saved to {self.checkpoint_path}")

    def save(self, file_path):
            """Save the model to the specified file path."""
            try:
                # Create the directory if it doesn't exist
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

                # Save the model and tokenizer using `save_pretrained` method
                self.model.save_pretrained(file_path)
                self.tokenizer.save_pretrained(file_path)

                self.logger.info(f"Model and tokenizer saved to {file_path}")
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
                raise


            #---------------------------------------------------------------------------------------------------------------------------------------------------------------
    def plot_metrics(self, evaluation=False):
        """
            Plots the training/validation losses and accuracies. If in evaluation mode, it plots evaluation losses and accuracies.
            :param evaluation: Boolean, if True the function plots evaluation data, else plots training and validation data.
            """
        self.logger.info('Plotting metrics.')  # New log entry

        plt.figure(figsize=(14, 6))

        # Determine the mode and prepare data accordingly
        if evaluation:
            # If we're in evaluation mode, we'll use the evaluation losses and accuracies
            if not hasattr(self, 'evaluation_losses') or not hasattr(self, 'evaluation_accuracies'):
                self.logger.info('No evaluation data to plot.')
                return

            epochs_range = range(1, len(self.evaluation_losses) + 1)
            losses = self.evaluation_losses
            accuracies = self.evaluation_accuracies
            loss_label = 'Evaluation Loss'
            accuracy_label = 'Evaluation Accuracy'

        else:
            # Ensure there's data to plot for training; if not, log a message and exit the function
            if not self.train_losses or not self.train_accuracies or not self.val_losses or not self.val_accuracies:
                self.logger.info('No training/validation data available to plot.')
                return
            # Check for the presence of training loss data before setting epochs_range
            if not self.train_losses:
                self.logger.warning('No training loss data available to plot.')
                return  # Exiting because there's no meaningful epochs_range to work with.

            # Proceeding with setting epochs_range since we have valid training loss data
            epochs_range = range(1, len(self.train_losses) + 1)
            losses = self.val_losses
            accuracies = self.val_accuracies
            loss_label = 'Validation Loss'
            accuracy_label = 'Validation Accuracy'

            # In non-evaluation mode, we plot training data as well
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, self.train_losses, label='Training Loss')
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, self.train_accuracies, label='Training Accuracy')

        # We need at least one epoch's worth of data to plot
        if len(losses) == 0 or len(accuracies) == 0:
            self.logger.warning('Not enough data to plot metrics.')
            return
        # Or we can check for sufficient data by these lines
        # if not losses or not accuracies:
        #     self.logger.warning('Not enough data to plot metrics.')
        #     return
        #------- plotting the metrics ------------#
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, losses, label=loss_label)
        plt.title('Loss over time')
        plt.legend(loc='best')

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, accuracies, label=accuracy_label)
        plt.title('Accuracy over time')
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        self.logger.info('Plotting confusion matrix.')  # New log entry

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(self.num_labels)
        plt.xticks(tick_marks, range(self.num_labels))
        plt.yticks(tick_marks, range(self.num_labels))

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
