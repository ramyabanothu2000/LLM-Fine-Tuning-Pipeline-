"""
config.py
---------
Dataclass-based training configuration for QLoRA fine-tuning.
Can be loaded from a YAML file or constructed in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class BitsAndBytesConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./outputs/finetuned"
    dataset_path: str = "./data/train.jsonl"

    # QLoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    bnb: BitsAndBytesConfig = field(default_factory=BitsAndBytesConfig)

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_seq_length: int = 2048
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 0.3
    group_by_length: bool = True
    packing: bool = False

    # Checkpointing
    save_steps: int = 100
    logging_steps: int = 25
    eval_steps: int = 100
    save_total_limit: int = 3

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "llm-finetuning"

    # HF Hub
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        lora_data = data.pop("lora", {})
        bnb_data = data.pop("bnb", {})

        config = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        config.lora = LoRAConfig(**lora_data) if lora_data else LoRAConfig()
        config.bnb = BitsAndBytesConfig(**bnb_data) if bnb_data else BitsAndBytesConfig()
        return config
