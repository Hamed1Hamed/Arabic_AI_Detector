## Overview
This project presents the first Arabic AI detector developed after the emergence of ChatGPT. It's designed to differentiate between Arabic synthetic ChatGPT texts and human-written texts, addressing critical gaps in current AI detection systems in dealing with this language. This code is customizable to adapt to any language. The only thing that needed to be adjusted is the Dediacritization Layer, which is designed specifically for the Arabic language.

## Our Design
- **Customizable:** Easily configurable via the `config.json` file containing all necessary hyperparameters.
- **Inference Module:** Utilize `ModelEvaluator.py`  if you want to investigate the performance for inference, leveraging the best model weights and Testing dataset of your choice.
## Installation of Dependencies
## Usage Instructions
pip install -r requirements.txt
### Configuration
1. **Modify Configurations:** Edit the `config.json` file to set the model hyperparameters as per your requirements.

### Model Weights
1. **Download Models:** Access the `Links To Experiments.txt` in the `Results` folder to download the model weights.
2. **Place Weights:** Ensure the downloaded weights are placed in the `check_point` folder for the inference module to function correctly.

### Dataset Preparation
1. **File Naming:** Edit the prefix of the dataset name. Our datasets are prefixed with labels like `customTraining`, `largeTraining`, etc. Rename these to `Training.csv`, `Validation.csv`, and `Testing.csv` for operational compatibility, and they have prefixed for readability purposes and organization within the dataset.
