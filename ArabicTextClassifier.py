from transformers import XLMRobertaForSequenceClassification
from torch.optim import AdamW

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
from transformers import XLMRobertaModel
import logging
import json
from transformers import AutoModel
import torch
import torch.nn as nn


logging.basicConfig(filename='classifier.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class CustomClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(CustomClassifierHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.ai_indicator_layer = nn.Linear(1, hidden_size)
        self.out_proj = nn.Linear(hidden_size * 2, num_labels)  # Adjusted size

    def forward(self, transformer_output, ai_indicator):
        x = self.dense(transformer_output)
        x = self.dropout(x)
        ai_indicator_processed = self.ai_indicator_layer(ai_indicator)
        concat = torch.cat((x, ai_indicator_processed), dim=-1)  # Exclude char_count_feature
        logits = self.out_proj(concat)
        return logits





class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.transformer = XLMRobertaModel.from_pretrained(model_name)
        self.custom_head = CustomClassifierHead(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, ai_indicator):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = transformer_outputs.last_hidden_state[:, 0, :]
        final_logits = self.custom_head(hidden_state, ai_indicator)
        return final_logits



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

    def forward(self, input_ids, attention_mask, ai_indicator):
        return self.model(input_ids, attention_mask, ai_indicator)

    # ----------------------------------------------------------------Training and Evaluation Functions-------------------------------------------------------------
    def load_best_model(self):
        model_path = os.path.join(self.checkpoint_path, "best_model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.logger.info(f"Loaded best model from {model_path}")
        else:
            self.logger.info("No best model checkpoint found.")

    def train(self, train_loader, val_loader, test_loader, start_epoch=0):
        best_val_loss = float('inf')
        epochs_without_improvement = 0  # Counter for early stopping
        final_train_accuracy = 0  # Initialize variable to store final training accuracy

        if not isinstance(train_loader, DataLoader) or not isinstance(val_loader, DataLoader) or not isinstance(
                test_loader, DataLoader):
            raise TypeError(
                "train_loader, val_loader, and test_loader must be DataLoader instances.")
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

            for batch in progress_bar:
                inputs, ai_indicator, labels = batch  # Unpack the ai_indicator tensor
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                ai_indicator = ai_indicator.to(self.device)  # Move ai_indicator to the correct device
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    ai_indicator=ai_indicator,
                )

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
            # Update final_train_accuracy with the latest training accuracy
            final_train_accuracy = correct_train_preds / max(total_train_preds, 1)

            # Validation step after each epoch
            avg_val_loss, val_accuracy = self.evaluate(val_loader, 'validation')

            # Scheduler step with the validation loss


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
            # Update final_train_accuracy with the latest training accuracy
            final_train_accuracy = correct_train_preds / max(total_train_preds, 1)

        self.logger.info('Training process completed. Starting testing on the test set.')

        # Testing on the test set after all epochs
        avg_test_loss, final_test_accuracy = self.evaluate(test_loader, 'testing')
        self.logger.info(f"Test Loss: {avg_test_loss}, Test Accuracy: {final_test_accuracy}")
        return final_train_accuracy, final_test_accuracy

    def evaluate(self, data_loader, context='validation'):
        assert context in ['validation', 'testing'], "Context must be either 'validation' or 'testing'"

        self.model.eval()
        self.model.to(self.device)
        total_loss = 0
        y_true = []
        y_pred = []
        correct_preds = 0
        total_preds = 0
        loss_fn = torch.nn.CrossEntropyLoss()

        # Storage for evaluation metrics if they don't already exist
        if not hasattr(self, f'evaluation_accuracies_{context}'):
            setattr(self, f'evaluation_accuracies_{context}', [])

        progress_bar = tqdm(data_loader, desc=f"Evaluating ({context})", leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                inputs, ai_indicator, labels = batch

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                ai_indicator = ai_indicator.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    ai_indicator=ai_indicator,
                )

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

                correct_preds += (predictions == labels).sum().item()
                total_preds += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_preds / total_preds

        # Logging metrics
        self.logger.info(f"{context.capitalize()} Accuracy: {accuracy}")
        self.logger.info(f"{context.capitalize()} Precision: {precision_score(y_true, y_pred, zero_division=0)}")
        self.logger.info(f"{context.capitalize()} Recall: {recall_score(y_true, y_pred, zero_division=0)}")
        self.logger.info(f"{context.capitalize()} F1 Score: {f1_score(y_true, y_pred, zero_division=0)}")
        self.logger.info(f"{context.capitalize()} AUC-ROC: {roc_auc_score(y_true, y_pred)}")

        # Append to the evaluation metrics lists
        getattr(self, f'evaluation_accuracies_{context}').append(accuracy)

        self.plot_confusion_matrix(y_true, y_pred, context)

        return avg_loss, accuracy

    # save()	Saves the model and tokenizer at a specified file path.
    # save_best_model()	Saves the best model and tokenizer based on a specific metric.
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
        # ------- plotting the metrics ------------#
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

    def plot_confusion_matrix(self, y_true, y_pred, context='validation'):
        assert context in ['validation', 'testing'], "Context must be either 'validation' or 'testing'"
        self.logger.info(f'Plotting confusion matrix for {context}.')  # Log entry

        # Define the labels for the confusion matrix
        classes = ['AI-generated', 'Human-written']

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate sensitivity and specificity
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1, 1] + cm[1, 0] > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0

        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(
            f'{context.capitalize()} Confusion Matrix\nSensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
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


    def plot_final_accuracies(final_train_accuracy, final_test_accuracy):
        # Ensure accuracies are provided
        if final_train_accuracy is None or final_test_accuracy is None:
            print("Final training or testing accuracy not provided.")
            return

        # Assuming you have a list of accuracies per epoch for training and testing
        # Replace the following lists with your actual data
        epochs = list(range(len(final_train_accuracy)))  # Replace with your range of epochs
        train_accuracies = final_train_accuracy  # Replace with your training accuracy data
        test_accuracies = final_test_accuracy  # Replace with your testing accuracy data

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_accuracies, label='train', color='blue')
        plt.plot(epochs, test_accuracies, label='test', color='green')

        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('model accuracy')
        plt.legend()
        plt.show()



