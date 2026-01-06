"""
Evaluation metrics for various LLM finetuning tasks.

This module provides comprehensive evaluation metrics for generative tasks,
classification, and question answering.
"""

import numpy as np
from typing import Dict, List, Union, Optional
from collections import defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GENERATIVE TASKS METRICS
# ============================================================================

def compute_generation_metrics(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = ["bleu", "rouge", "bertscore"]
) -> Dict[str, float]:
    """
    Compute metrics for text generation tasks.

    Args:
        predictions: List of generated texts
        references: List of reference texts
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric scores
    """
    results = {}

    try:
        if "bleu" in metrics:
            from evaluate import load
            bleu = load("bleu")
            # BLEU expects references as list of lists
            refs = [[ref] for ref in references]
            bleu_score = bleu.compute(predictions=predictions, references=refs)
            results["bleu"] = bleu_score["bleu"]
            results["bleu_1"] = bleu_score["precisions"][0] if len(bleu_score["precisions"]) > 0 else 0
            results["bleu_2"] = bleu_score["precisions"][1] if len(bleu_score["precisions"]) > 1 else 0

    except Exception as e:
        print(f"BLEU computation failed: {e}")
        results["bleu"] = 0.0

    try:
        if "rouge" in metrics:
            from evaluate import load
            rouge = load("rouge")
            rouge_scores = rouge.compute(predictions=predictions, references=references)
            results["rouge1"] = rouge_scores["rouge1"]
            results["rouge2"] = rouge_scores["rouge2"]
            results["rougeL"] = rouge_scores["rougeL"]

    except Exception as e:
        print(f"ROUGE computation failed: {e}")

    try:
        if "bertscore" in metrics:
            from evaluate import load
            bertscore = load("bertscore")
            bert_scores = bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                model_type="distilbert-base-uncased"  # Faster model
            )
            results["bertscore_f1"] = np.mean(bert_scores["f1"])
            results["bertscore_precision"] = np.mean(bert_scores["precision"])
            results["bertscore_recall"] = np.mean(bert_scores["recall"])

    except Exception as e:
        print(f"BERTScore computation failed: {e}")

    try:
        if "meteor" in metrics:
            from evaluate import load
            meteor = load("meteor")
            meteor_score = meteor.compute(predictions=predictions, references=references)
            results["meteor"] = meteor_score["meteor"]

    except Exception as e:
        print(f"METEOR computation failed: {e}")

    # Diversity metrics
    if "diversity" in metrics:
        diversity_scores = compute_diversity_metrics(predictions)
        results.update(diversity_scores)

    return results


def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity metrics (distinct-1, distinct-2).

    Args:
        texts: List of generated texts

    Returns:
        Dictionary with diversity scores
    """
    all_unigrams = []
    all_bigrams = []

    for text in texts:
        tokens = text.lower().split()

        # Unigrams
        all_unigrams.extend(tokens)

        # Bigrams
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        all_bigrams.extend(bigrams)

    # Distinct-n: ratio of unique n-grams to total n-grams
    distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2
    }


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    device: str = "cuda"
) -> float:
    """
    Compute perplexity on a list of texts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of texts
        max_length: Maximum sequence length
        device: Device to use

    Returns:
        Average perplexity
    """
    import torch

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(device)

            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss

            total_loss += loss.item() * encodings["input_ids"].size(1)
            total_tokens += encodings["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def compute_classification_metrics(
    predictions: Union[List[int], np.ndarray],
    labels: Union[List[int], np.ndarray],
    num_classes: Optional[int] = None,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes (auto-detected if None)
        average: Averaging method (weighted, macro, micro)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )

    predictions = np.array(predictions)
    labels = np.array(labels)

    results = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average=average, zero_division=0),
        "recall": recall_score(labels, predictions, average=average, zero_division=0),
        "f1": f1_score(labels, predictions, average=average, zero_division=0)
    }

    # Per-class metrics
    if num_classes:
        for avg_type in ["macro", "micro", "weighted"]:
            results[f"f1_{avg_type}"] = f1_score(labels, predictions, average=avg_type, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    results["confusion_matrix"] = cm.tolist()

    return results


def print_classification_report(
    predictions: Union[List[int], np.ndarray],
    labels: Union[List[int], np.ndarray],
    target_names: Optional[List[str]] = None
):
    """
    Print detailed classification report.

    Args:
        predictions: Predicted labels
        labels: True labels
        target_names: Class names
    """
    from sklearn.metrics import classification_report

    print("\n=== Classification Report ===")
    print(classification_report(
        labels,
        predictions,
        target_names=target_names,
        zero_division=0
    ))


# ============================================================================
# QUESTION ANSWERING METRICS
# ============================================================================

def compute_qa_metrics(
    predictions: List[Dict],
    references: List[Dict]
) -> Dict[str, float]:
    """
    Compute question answering metrics (Exact Match and F1).

    Args:
        predictions: List of dicts with 'prediction_text' and 'id'
        references: List of dicts with 'answers' and 'id'

    Returns:
        Dictionary with EM and F1 scores
    """
    from evaluate import load

    squad_metric = load("squad")

    # Format for SQuAD metric
    formatted_predictions = [
        {"id": p["id"], "prediction_text": p["prediction_text"]}
        for p in predictions
    ]

    formatted_references = [
        {"id": r["id"], "answers": r["answers"]}
        for r in references
    ]

    results = squad_metric.compute(
        predictions=formatted_predictions,
        references=formatted_references
    )

    return {
        "exact_match": results["exact_match"],
        "f1": results["f1"]
    }


def normalize_answer(s: str) -> str:
    """Normalize answer string for QA evaluation."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match between prediction and ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    # If either is empty
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# ============================================================================
# EVALUATION SUITE
# ============================================================================

class EvaluationSuite:
    """Comprehensive evaluation suite for different task types."""

    def __init__(self, task_type: str):
        """
        Initialize evaluation suite.

        Args:
            task_type: Type of task (generation, classification, qa)
        """
        self.task_type = task_type
        self.results = {}

    def evaluate(
        self,
        predictions: Union[List[str], List[int], np.ndarray],
        references: Union[List[str], List[int], np.ndarray],
        **kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation based on task type.

        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional arguments for specific metrics

        Returns:
            Dictionary of evaluation results
        """
        if self.task_type == "generation":
            self.results = compute_generation_metrics(
                predictions,
                references,
                metrics=kwargs.get("metrics", ["bleu", "rouge"])
            )

        elif self.task_type == "classification":
            self.results = compute_classification_metrics(
                predictions,
                references,
                num_classes=kwargs.get("num_classes"),
                average=kwargs.get("average", "weighted")
            )

        elif self.task_type == "qa":
            self.results = compute_qa_metrics(predictions, references)

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        return self.results

    def print_results(self):
        """Print evaluation results in a formatted way."""
        print("\n=== Evaluation Results ===")
        print(f"Task Type: {self.task_type}\n")

        for metric, value in self.results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            elif isinstance(value, list):
                print(f"  {metric}: [matrix]")
            else:
                print(f"  {metric}: {value}")

    def save_results(self, output_path: str):
        """Save evaluation results to JSON file."""
        import json

        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                results_to_save[key] = value.tolist()
            else:
                results_to_save[key] = value

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"Results saved to: {output_path}")


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(
    base_results: Dict[str, float],
    finetuned_results: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Compare base model vs finetuned model results.

    Args:
        base_results: Base model evaluation results
        finetuned_results: Finetuned model results

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "base": base_results,
        "finetuned": finetuned_results,
        "improvement": {}
    }

    # Calculate improvements
    for metric in base_results.keys():
        if metric in finetuned_results and isinstance(base_results[metric], (int, float)):
            base_val = base_results[metric]
            finetuned_val = finetuned_results[metric]

            if base_val != 0:
                improvement_pct = ((finetuned_val - base_val) / base_val) * 100
            else:
                improvement_pct = 0

            comparison["improvement"][metric] = {
                "absolute": finetuned_val - base_val,
                "percentage": improvement_pct
            }

    return comparison


def print_comparison(comparison: Dict):
    """Print model comparison in a formatted table."""
    print("\n=== Model Comparison ===\n")

    print(f"{'Metric':<20} {'Base':<12} {'Finetuned':<12} {'Δ Absolute':<12} {'Δ %':<12}")
    print("-" * 80)

    for metric in comparison["base"].keys():
        if isinstance(comparison["base"][metric], (int, float)):
            base = comparison["base"][metric]
            finetuned = comparison["finetuned"].get(metric, 0)

            if metric in comparison["improvement"]:
                abs_imp = comparison["improvement"][metric]["absolute"]
                pct_imp = comparison["improvement"][metric]["percentage"]

                print(f"{metric:<20} {base:<12.4f} {finetuned:<12.4f} {abs_imp:+<12.4f} {pct_imp:+<12.2f}%")
