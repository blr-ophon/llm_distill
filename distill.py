import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluate import load
from trainer_distill import TrainingArgumentsDistill, TrainerDistill


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def format_sample(sample):
    if sample["input"]:
        text = (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    return {"text": text}


def tokenize_sample(sample, tokenizer):
    tokens = tokenizer(
        sample["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def student_init(student_ckpt, config):
    model = torch.compile(AutoModelForCausalLM.from_pretrained(student_ckpt, config=config))
    return model.to(device)


def distill():
    teacher_ckpt = "Qwen/Qwen2-1.5B"
    student_ckpt = "Qwen/Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(student_ckpt)

    # Prepare and tokenize dataset
    dataset = load_dataset("yahma/alpaca-cleaned")["train"].train_test_split(train_size=0.8)
    column_names = ['text', 'instruction', 'output', 'input']

    train_dataset = dataset["train"].map(format_sample)
    train_dataset = train_dataset.map(lambda a: tokenize_sample(a, tokenizer), batched=True)
    train_dataset = train_dataset.remove_columns(column_names)
    test_dataset = dataset["test"].map(format_sample)
    test_dataset = test_dataset.map(lambda a: tokenize_sample(a, tokenizer), batched=True)
    test_dataset = test_dataset.remove_columns(column_names)

    #print(train_dataset[0])
    #return

    # Training arguments
    training_args = TrainingArgumentsDistill(
        output_dir="distill_qwen",
        overwrite_output_dir=True,

        evaluation_strategy="steps",
        eval_steps=500,                    # Evaluate every 500 steps
        save_steps=500,                    # Save checkpoint every 500 steps
        logging_steps=100,                 # Log metrics every 100 steps

        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        warmup_steps=500,                  # Helps stabilize training at start
        weight_decay=0.01,
        learning_rate=3e-4,

        save_total_limit=2,                # Only keep last 2 checkpoints
        logging_dir="./logs",

        report_to="none",                  # Disable wandb/huggingface logging
        remove_unused_columns=False,
    )

    # Trainer
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_ckpt).to(device)
    student_config = AutoConfig.from_pretrained(student_ckpt)

    trainer = TrainerDistill(
        model_init=(lambda x: student_init(student_ckpt, student_config)),
        teacher_model=teacher_model, args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    return


if __name__ == """__main__""":
    distill()
