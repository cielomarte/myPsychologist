# preprocessing.py
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def load_and_split_dataset(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Splitting the dataset
    train_test_split = dataset["train"].train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_validation_split = train_test_split["test"].train_test_split(test_size=0.5)
    validation_dataset = test_validation_split["train"]
    test_dataset = test_validation_split["test"]

    return train_dataset, validation_dataset, test_dataset

def tokenize_and_format_dataset(train_dataset, validation_dataset, test_dataset, model_checkpoint):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["Context"], examples["Response"], truncation=True, padding="max_length", max_length=512)

    # Apply tokenization
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_validation = validation_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Format the datasets
    def format_dataset(dataset):
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return dataset

    formatted_train = format_dataset(tokenized_train)
    formatted_validation = format_dataset(tokenized_validation)
    formatted_test = format_dataset(tokenized_test)

    return formatted_train, formatted_validation, formatted_test

