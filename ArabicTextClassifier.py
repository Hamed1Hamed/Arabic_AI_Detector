from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizer
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
from transformers import AutoModel
import torch
import torch.nn as nn

class CustomClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(CustomClassifierHead, self).__init__()
        # The transformer model output size (e.g., 768 for BERT-base)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        # Additional input for the binary feature
        self.binary_feature = nn.Linear(1, hidden_size)
        self.out_proj = nn.Linear(hidden_size * 3, num_labels)  # Change input size to include Char_Count

    def forward(self, transformer_output, binary_feature, char_count_feature):
        # Assuming transformer_output is [batch_size, hidden_size]
        x = self.dense(transformer_output)
        x = self.dropout(x)

        # Assuming binary_feature is [batch_size, 1]
        binary_feature = self.binary_feature(binary_feature)  # Now [batch_size, hidden_size]

        # Assuming char_count_feature is [batch_size, 1]
        char_count_feature = self.binary_feature(char_count_feature)  # Now [batch_size, hidden_size]

        # Concatenate the transformer output, binary feature, and char_count feature
        concat = torch.cat((x, binary_feature, char_count_feature), dim=-1)  # Ensure the dimension is correct

        logits = self.out_proj(concat)
        return logits


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        # Load the pre-trained model
        self.transformer = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        hidden_size = self.transformer.config.hidden_size
        # Additional input for the binary feature
        self.binary_feature = nn.Linear(1, hidden_size)
        self.char_count_feature = nn.Linear(1, hidden_size)  # Add the Char_Count feature layer

    def forward(self, input_ids, attention_mask, ai_indicator, char_count_feature):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = transformer_outputs.logits

        # Assuming binary_feature is [batch_size, 1]
        binary_feature = self.binary_feature(ai_indicator)
        char_count_feature = self.char_count_feature(char_count_feature)  # Process Char_Count feature

        # Combine transformer output, binary feature, and char_count feature
        combined = transformer_outputs.last_hidden_state[:, 0, :] + binary_feature + char_count_feature

        # You can further process the combined representation before classification if necessary

        return logits




class ArabicTextClassifier(nn.Module):
    def __init__(self, model_name, num_labels, learning_rate, epochs, checkpoint_path):
        super(ArabicTextClassifier, self).__init__()
        self.model = CustomModel(model_name, num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.logger = logging.getLogger(__name__)
        self.checkpoint_path = checkpoint_path
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ai_indicator, char_count_feature):
        return self.model(input_ids, attention_mask, ai_indicator, char_count_feature)



    #----------------------------------------------------------------Training and Evaluation Functions-------------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader, char_count_loader, start_epoch=0):
        best_val_loss = float('inf')
        epochs_without_improvement = 0  # Counter for early stopping

        if not isinstance(train_loader, DataLoader) or not isinstance(val_loader, DataLoader) or not isinstance(
                test_loader, DataLoader) or not isinstance(char_count_loader, DataLoader):
            raise TypeError(
                "train_loader, val_loader, test_loader, and char_count_loader must be DataLoader instances.")
        self.model.to(self.device)  # Make sure this is done before training begins
        # Initialize the scheduler
        scheduler = CosineAnnealingLR(self.optimizer, T_max=len(train_loader) * self.epochs, eta_min=0)
        self.logger.info('Training process started.')

        for epoch in range(start_epoch, self.epochs):
            self.model.train()  # Begin training
            total_train_loss = 0
            correct_train_preds = 0
            total_train_preds = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)

            for batch, char_count_batch in zip(progress_bar, char_count_loader):
                inputs, ai_indicator, labels = batch  # Unpack the ai_indicator tensor
                char_count_feature = char_count_batch  # Extract Char_Count feature from char_count_loader

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                ai_indicator = ai_indicator.to(self.device)  # Move ai_indicator to the correct device
                char_count_feature = char_count_feature.to(self.device)  # Move Char_Count feature to the correct device
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                    ai_indicator=ai_indicator, char_count_feature=char_count_feature)
                # loss = torch.nn.functional.cross_entropy(logits, labels)  # Calculate loss
                loss = self.loss_fn(logits, labels)

                loss_value = loss.item()
                total_train_loss += loss_value
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                # Update the scheduler after each batch
                scheduler.step()

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

            # Validation step after each epoch
            avg_val_loss, val_accuracy = self.evaluate(val_loader)

            # Scheduler step with the validation loss
            scheduler.step(avg_val_loss)

            # Checkpoint if this is the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                self.save_best_model()  # Save only the best model

            # Record metrics for each epoch
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)

        self.logger.info('Training process completed. Starting testing on the test set.')

        # Testing on the test set after all epochs
        avg_test_loss, test_accuracy = self.evaluate(test_loader)
        self.logger.info(f"Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}")

    def evaluate(self, val_loader, char_count_loader):
        self.model.eval()
        self.model.to(self.device)  # Make sure this is done before evaluation
        total_val_loss = 0
        y_true = []
        y_pred = []
        correct_preds = 0
        total_preds = 0
        loss_fn = torch.nn.CrossEntropyLoss()

        # Storage for evaluation metrics if they don't already exist
        if not hasattr(self, 'evaluation_accuracies'):
            self.evaluation_accuracies = []

        progress_bar = tqdm(zip(val_loader, char_count_loader), desc="Evaluating", leave=True)

        with torch.no_grad():
            for (batch, char_count_batch) in progress_bar:
                inputs, ai_indicator, labels = batch
                char_count_feature = char_count_batch  # Extract Char_Count feature from char_count_loader

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                ai_indicator = ai_indicator.to(self.device)  # Move ai_indicator to the correct device
                char_count_feature = char_count_feature.to(self.device)  # Move Char_Count feature to the correct device
                labels = labels.to(self.device)

                # Forward pass through the model to get logits
                logits = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                    ai_indicator=ai_indicator,
                                    char_count_feature=char_count_feature)

                # Compute loss using the logits and the labels
                loss = self.loss_fn(logits, labels)
                total_val_loss += loss.item()

                # Get the predictions from the logits
                predictions = torch.argmax(logits, dim=-1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

                correct_preds += (predictions == labels).sum().item()
                total_preds += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
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

        return avg_val_loss, val_accuracy

    #save()	Saves the model and tokenizer at a specified file path.
#save_best_model()	Saves the best model and tokenizer based on a specific metric.
    def save_best_model(self):
        # Save the best model without including the epoch in the filename
        model_save_path = os.path.join(self.checkpoint_path, "best_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(f"Best model saved to {model_save_path}")

    def save(self, file_path):
        """Save the model state dictionary to the specified file path."""
        try:
            # Create the directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            # Save the model state dictionary
            model_save_path = os.path.join(file_path, "model_state_dict.pt")
            torch.save(self.model.state_dict(), model_save_path)

            self.logger.info(f"Model state dictionary saved to {model_save_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

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
        self.logger.info('Plotting confusion matrix.')  # Log entry

        # Define the labels for the confusion matrix
        classes = ['AI-generated', 'Human-written']

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate sensitivity and specificity
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix\nSensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Label the axes
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        # Loop over the data locations to create text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()

