# training.py
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments

def train_model(formatted_train, formatted_validation, model_checkpoint, output_dir):
    # Load the pre-trained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train,
        eval_dataset=formatted_validation
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(f"{output_dir}/fine_tuned_model")
