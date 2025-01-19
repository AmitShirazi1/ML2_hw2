# Project Title
**HW2: Machine Learning Analysis**

## Overview
This project focuses on building and evaluating a machine learning model using PyTorch to classify emotions based on text data. It explores the concepts of generalization, overfitting, and classification metrics. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Features
- Data preprocessing with tokenization and cleaning.
- Implementation of a PyTorch-based neural network for emotion classification.
- Use of confusion matrix and label distribution visualization for model evaluation.
- Exploration of training and testing datasets for insights into label distributions and tweet lengths.

## Requirements
The project uses the following libraries and frameworks:
- Python
- PyTorch
- pandas
- NumPy
- Matplotlib
- scikit-learn
- spaCy

## File Structure
- `HW2_319044434_314779166.ipynb`: The main Jupyter Notebook containing all the code and outputs.

## Setup Instructions
1. Install the required libraries:
   ```bash
   pip install torch torchvision pandas numpy matplotlib scikit-learn spacy
   ```
2. Download the spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
3. Prepare your data files (training and testing CSV files).

## Usage
1. Load and preprocess data:
   - The data preprocessing steps include text cleaning, tokenization using spaCy, and encoding with a vocabulary.
2. Train the neural network model:
   - Utilize the provided `train_and_evaluate` function to train the model and validate its performance on test data.
3. Visualize results:
   - Use the built-in functions to plot confusion matrices and compare training vs testing label distributions.

## Key Functions
- `load_data(train_file, test_file)`: Loads the training and testing datasets.
- `explore_data(train_data, test_data)`: Explores datasets for insights such as label distributions and tweet lengths.
- `preprocess_data(data, vocab=None)`: Cleans and tokenizes the text data.
- `prepare_data_loaders(...)`: Prepares PyTorch DataLoader objects for model training and testing.
- `train_and_evaluate(...)`: Trains the model and evaluates its performance.
- `plot_confusion_matrix(...)`: Visualizes the confusion matrix for model predictions.

## Visualization
The notebook includes multiple visualizations:
- Bar charts comparing training and testing label distributions.
- Histograms showing tweet length distributions.
- Confusion matrices displaying model performance.

## Notes
- The datasets are expected to have columns like `content` (text data) and `emotion` (labels).
- The neural network model is implemented using PyTorch and supports batch processing.

## Results
- The project demonstrates how preprocessing and model evaluation are critical for effective machine learning solutions.
- Insights from label distributions and text lengths help refine preprocessing strategies.
