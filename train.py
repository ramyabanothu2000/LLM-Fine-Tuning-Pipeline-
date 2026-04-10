#!/usr/bin/env python3
"""
train.py — Training entrypoint for QLoRA fine-tuning.

Usage:
    python scripts/train.py --config configs/llama2_7b_qlora.yaml
    python scripts/train.py --config configs/mistral_7b_qlora.yaml --data_path ./data/my_data.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import TrainingConfig
from src.training.trainer import QLoRAFinetuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA LLM Fine-tuning")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data_path", help="Override dataset path from config")
    parser.add_argument("--output_dir", help="Override output directory from config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig.from_yaml(args.config)

    if args.data_path:
        config.dataset_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir

    print(f"\n{'='*60}")
    print(f"  Model       : {config.model_name}")
    print(f"  Dataset     : {config.dataset_path}")
    print(f"  Output      : {config.output_dir}")
    print(f"  LoRA r      : {config.lora.r}")
    print(f"  Epochs      : {config.num_train_epochs}")
    print(f"  Batch size  : {config.per_device_train_batch_size}")
    print(f"  LR          : {config.learning_rate}")
    print(f"{'='*60}\n")

    tuner = QLoRAFinetuner(config)
    tuner.train()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
