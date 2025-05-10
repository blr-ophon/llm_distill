# Third party imports
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from liger_kernel.transformers import AutoLigerKernelForCausalLM
# Local imports
from trainer_distill import TrainingArgumentsDistill, TrainerDistill
from peft import PeftModel, PeftConfig, get_peft_model


def distill():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load teacher model
    teacher_ckpt = "Qwen/Qwen2-0.5B"
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_ckpt)
    teacher_model = teacher_model.to(device)

    # Load student model
    student_ckpt = "jtromero/qwen2-0.5b-lora-single-device"
    student_model = student_init_lora(student_ckpt, custom_kernel=False,
                                      compile_model=True)
    student_model = student_model.to(device)

    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(student_ckpt)
    dataset = load_dataset("yahma/alpaca-cleaned")["train"].train_test_split(train_size=0.8)
    dataset = tokenize_dataset(dataset, tokenizer)

    # Training arguments
    training_args = TrainingArgumentsDistill(
        output_dir="distill_qwen",
        overwrite_output_dir=True,

        eval_strategy="steps",
        eval_steps=500,                    # Evaluate every n steps
        save_steps=500,                    # Save checkpoint every n steps
        logging_steps=100,                 # Log metrics every n steps

        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=3e-4,

        save_total_limit=2,                # Only keep last 2 checkpoints
        logging_dir="./logs",

        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = TrainerDistill(
        model=student_model,
        teacher_model=teacher_model, args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer
    )
    trainer.train()


def student_init(student_ckpt, custom_kernel=False, compile_model=False):
    config = AutoConfig.from_pretrained(student_ckpt)

    # Load base model
    if custom_kernel:
        model = AutoLigerKernelForCausalLM.from_pretrained(student_ckpt, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(student_ckpt, config=config)

    # Compile for performance
    model = torch.compile(model) if compile_model else model

    return model


def student_init_lora(student_ckpt, custom_kernel=False, compile_model=False):
    peft_config = PeftConfig.from_pretrained(student_ckpt)

    # Load base model
    if custom_kernel:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path
        )

    # Compile for performance
    model = torch.compile(model) if compile_model else model

    # Load LoRA adapter
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model


def generate_text_column(sample):
    """
    Generate prompt template recommended in the dataset card for the
    alpaca cleaned dataset.
    """
    if sample["input"]:
        text = (
            f"Below is an instruction that describes a task, paired with an "
            f"input that provides further context. Write a response that "
            f"appropriately completes the request.\n\n"
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    else:
        text = (
            f"Below is an instruction that describes a task. Write a response "
            f"that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    return {"text": text}


def tokenize_text(sample, tokenizer):
    tokens = tokenizer(
        sample["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def tokenize_dataset(dataset, tokenizer):
    """
    Prepare and tokenize dataset
    """
    try:
        dataset = load_from_disk("./tokenized_dataset")
    except:
        dataset = dataset.map(generate_text_column)
        dataset = dataset.map(lambda a: tokenize_text(a, tokenizer),
                              batched=True)
        # Remove unused columns
        column_names = ['text', 'instruction', 'output', 'input']
        dataset = dataset.remove_columns(column_names)
        dataset.save_to_disk("./tokenized_dataset")

    return dataset


if __name__ == """__main__""":
    distill()
