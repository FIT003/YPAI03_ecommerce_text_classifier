# e-Commerce Text Classifier

## Description
This repository contains code for an e-Commerce Text Classifier. The code is designed to classify text data from the e-commerce domain into different categories. The classifier is trained using a dataset obtained from Kaggle. Text dataset used have 4 categories - "Electronics", "Household", "Books" and "Clothing & Accessories", which almost cover 80% of any E-commerce website.

## Installation
To use this code, please follow the steps below:

1. Clone the repository to your local machine using the following command

```
git clone [repository URL]
```

2. Navigate to the project directory:

```
cd [project directory]
```

3. Install the required dependencies by running the following command:

```
pip install pandas numpy tensorflow matplotlib scikit-learn
```

4. Download the dataset from URL provided in the **Credit**.

5. Place the downloaded CSV files in the project directory.

## Usage
Run the Jupyter notebook or Python script to execute the code.

The code performs the following steps:

1. Imports necessary packages.
2. Defines hyperparameters for the model.
- Loads the e-commerce text data from the CSV file.
- Performs data inspection and cleaning.
- Converts labels into integers using label encoding.
- Splits the dataset into training and testing sets.
- Tokenizes the training features.
- Pads and truncates the tokenized sequences.
- Builds a bidirectional LSTM-based neural network model.
- Compiles and trains the model.
- Evaluates the model's performance using loss and accuracy metrics.
- Deploys the model by predicting the category of a new input text.
- Saves the trained model and tokenizer for future use.
3. Customize the code as needed, such as changing hyperparameters, adjusting model architecture, or modifying the data preprocessing steps.

## Outputs

- model architecture
![model_architecture](https://github.com/FIT003/YPAI03_ecommerce_text_classifier/assets/97938451/3f6619fe-26a4-41b2-b87e-f930aef30c13)

- model training
![model_training](https://github.com/FIT003/YPAI03_ecommerce_text_classifier/assets/97938451/c3d2983e-eb5c-49e7-ba27-48ba9e98cc38)

- loss
![loss](https://github.com/FIT003/YPAI03_ecommerce_text_classifier/assets/97938451/19d06599-dfbe-48f9-b793-74d64c49f8c9)

- accuracy
![accuracy](https://github.com/FIT003/YPAI03_ecommerce_text_classifier/assets/97938451/3dd87ec3-294a-4852-9948-e7a947372c91)

## Credits
URL: https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification/code


