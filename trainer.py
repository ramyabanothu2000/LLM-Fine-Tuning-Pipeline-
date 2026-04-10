"""
trainer.py
----------
QLoRA fine-tuning using Hugging Face TRL SFTTrainer + PEFT.
Supports LLaMA 2 and Mistral architectures.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from .config import TrainingConfig

logger = logging.getLogger(__name__)


INSTRUCTION_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


class QLoRAFinetuner:
    """
    Fine-tunes a causal LLM using QLoRA via TRL SFTTrainer.

    Parameters
    ----------
    config : TrainingConfig
        All hyperparameters and paths.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self):
        logger.info("Loading base model: %s", self.config.model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.bnb.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=self.config.bnb.bnb_4bit_use_double_quant,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        self.model = model
        self.tokenizer = tokenizer

    def _apply_lora(self):
        lora_cfg = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _load_dataset(self) -> Dataset:
        logger.info("Loading dataset from %s", self.config.dataset_path)
        ds = load_dataset("json", data_files={"train": self.config.dataset_path})
        return ds["train"]

    def _format_prompt(self, example: dict) -> dict:
        text = INSTRUCTION_TEMPLATE.format(
            instruction=example.get("instruction", ""),
            input=example.get("input", ""),
            output=example.get("output", ""),
        )
        return {"text": text}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        self._load_model_and_tokenizer()
        self._apply_lora()

        dataset = self._load_dataset()
        dataset = dataset.map(self._format_prompt, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_grad_norm=self.config.max_grad_norm,
            group_by_length=self.config.group_by_length,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            report_to="mlflow",
            run_name=f"{Path(self.config.model_name).name}-qlora",
        )

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
        )

        logger.info("Starting training...")
        with mlflow.start_run():
            mlflow.log_params({
                "model_name": self.config.model_name,
                "lora_r": self.config.lora.r,
                "lora_alpha": self.config.lora.alpha,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.num_train_epochs,
                "batch_size": self.config.per_device_train_batch_size,
                "max_seq_length": self.config.max_seq_length,
                "quant_type": self.config.bnb.bnb_4bit_quant_type,
            })
            trainer.train()
            trainer.save_model(self.config.output_dir)
            mlflow.log_artifacts(self.config.output_dir, artifact_path="model")

        logger.info("Training complete. Model saved to %s", self.config.output_dir)
