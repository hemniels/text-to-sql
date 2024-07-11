# Text-to-SQL Transformer Model

This repository contains a project that trains a transformer model to convert natural language questions into SQL queries using the Hugging Face Transformers library. The T5 model (Text-to-Text Transfer Transformer) is used in this example.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

This project demonstrates how to use a transformer model to translate natural language questions into SQL queries. The project is structured into multiple files for clarity and modularity, leveraging the Hugging Face `transformers` library and `datasets` library to load, train, and evaluate the model.

## Requirements

To install the required packages, run:
```bash
pip install -r requirements.txt
pip install datasets
```
## Dataset
The dataset used for this project is pg-wikiSQL-sql-instructions-80k, available on Hugging Face.

## Training the Model
To train the model, run:
This script:
```bash
python train.py
```

Loads and preprocesses the dataset.
Initializes and fine-tunes the T5 model.
Saves the trained model and tokenizer.

## Making Predictions
To make predictions using the trained model, run:

This script:
```bash
python predict.py
```
Loads the trained model and tokenizer.
Converts a natural language question into an SQL query.

## Project Structure
.  
├── dataset.py          # Script to load and preprocess the dataset  
├── model.py            # Script to define the model and training process  
├── train.py            # Script to handle the training process  
├── predict.py          # Script to handle predictions  
├── requirements.txt    # File to list the dependencies  
└── README.md           # This readme file  