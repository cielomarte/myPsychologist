# main.py
from preprocessing import load_and_split_dataset, tokenize_and_format_dataset
from training import train_model

def main():
    # Define dataset and model checkpoint
    dataset_name = "Amod/mental_health_counseling_conversations"
    model_checkpoint = "LLaMA-2-7b-specific-checkpoint"
    output_dir = "./results"

    # Load and split the dataset
    train_dataset, validation_dataset, test_dataset = load_and_split_dataset(dataset_name)

    # Tokenize and format the dataset
    formatted_train, formatted_validation, formatted_test = tokenize_and_format_dataset(train_dataset, validation_dataset, test_dataset, model_checkpoint)

    # Train the model
    train_model(formatted_train, formatted_validation, model_checkpoint, output_dir)

if __name__ == "__main__":
    main()
