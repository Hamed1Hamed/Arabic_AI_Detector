
from itertools import product
from ArabicTextDataset import ArabicTextDataset
from transformers import AutoTokenizer
from ArabicTextClassifier import *
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

" This class is used to evaluate the model on the testing set."
class ModelEvaluator:
    def __init__(self, model_path, model_name, num_labels, device):
        # Set up logging
        logging.basicConfig(filename='classifier.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ModelEvaluator...")

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

        # Load model
        try:
            self.model = CustomModel(model_name, num_labels)
            self.model.load_state_dict(torch.load(model_path))
            self.device = device
            self.model.to(self.device)
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.logger.info("Model loaded for evaluation. The model is now evaluating the testing set using existing weights.")
            self.logger.info(f"Model loaded for evaluation. Loaded best model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model. Error: {e}")
            raise
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []

        progress_bar = tqdm(data_loader, desc="Evaluating (testing set)", leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                logits = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred)

        self.logger.info("Testing Evaluation Metrics:")
        self.logger.info(f"  - Average Loss: {avg_loss}")
        self.logger.info(f"  - Accuracy: {accuracy}")
        self.logger.info(f"  - Precision: {precision}")
        self.logger.info(f"  - Recall: {recall}")
        self.logger.info(f"  - F1 Score: {f1}")
        self.logger.info(f"  - AUC-ROC: {auc}")

        self.plot_confusion_matrix(y_true, y_pred)

        return avg_loss

    def plot_confusion_matrix(self, y_true, y_pred):
        classes = ['AI-generated', 'Human-written']
        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1, 1] + cm[1, 0] > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0

        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title(f'Testing Confusion Matrix\nSensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.tight_layout()
        plt.show()

# Function to run the evaluation independently
def run_evaluation():

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = ArabicTextDataset(tokenizer, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config['testing_batch_size'], shuffle=False)

    # Initialize Model Evaluator
    best_model_path = os.path.join(config['checkpoint_path'], "best_model.pt")
    model_evaluator = ModelEvaluator(best_model_path, model_name, num_labels=2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Run evaluation
    model_evaluator.evaluate(test_loader)

if __name__ == '__main__':
    run_evaluation()
