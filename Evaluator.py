"""
evaluator.py
------------
Evaluates fine-tuned LLMs on held-out test data.
Metrics: ROUGE-L, BLEU-4, F1 (token-level), BERTScore.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import torch
from bert_score import score as bert_score
from datasets import Dataset, load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

INSTRUCTION_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


class LLMEvaluator:
    """
    Runs inference on test set and computes evaluation metrics.

    Parameters
    ----------
    model_path : str
        Path to the saved PEFT adapter (or merged model directory).
    base_model_name : str
        HuggingFace model ID of the base model (needed for PEFT loading).
    """

    def __init__(self, model_path: str, base_model_name: str):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.pipe = self._load_pipeline()
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def _load_pipeline(self):
        logger.info("Loading model from %s", self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, self.model_path)
        model.eval()

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )

    def _generate(self, instruction: str, input_text: str) -> str:
        prompt = INSTRUCTION_TEMPLATE.format(
            instruction=instruction, input=input_text
        )
        output = self.pipe(prompt)[0]["generated_text"]
        # Extract only the Response portion
        if "### Response:" in output:
            output = output.split("### Response:")[-1].strip()
        return output

    def evaluate(self, test_path: str, mlflow_run: bool = True) -> Dict[str, float]:
        """
        Run evaluation on a JSONL test file.

        Expected format: {"instruction": ..., "input": ..., "output": ...}
        """
        ds = load_dataset("json", data_files={"test": test_path})["test"]

        predictions: List[str] = []
        references: List[str] = []

        for example in ds:
            pred = self._generate(example["instruction"], example.get("input", ""))
            predictions.append(pred)
            references.append(example["output"])

        metrics = self._compute_metrics(predictions, references)
        logger.info("Evaluation results: %s", metrics)

        if mlflow_run:
            with mlflow.start_run(run_name="evaluation"):
                mlflow.log_metrics(metrics)

        return metrics

    def _compute_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        # ROUGE-L
        rouge_scores = [
            self.rouge.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        avg_rouge = float(np.mean(rouge_scores))

        # Token-level F1
        f1_scores = [
            self._token_f1(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        avg_f1 = float(np.mean(f1_scores))

        # BLEU-4
        bleu = corpus_bleu(
            [[ref.split()] for ref in references],
            [pred.split() for pred in predictions],
            smoothing_function=SmoothingFunction().method1,
        )

        # BERTScore (batch)
        _, _, F = bert_score(predictions, references, lang="en", verbose=False)
        avg_bertscore = float(F.mean())

        return {
            "rouge_l": round(avg_rouge, 4),
            "token_f1": round(avg_f1, 4),
            "bleu_4": round(bleu, 4),
            "bertscore_f1": round(avg_bertscore, 4),
        }

    @staticmethod
    def _token_f1(prediction: str, reference: str) -> float:
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
