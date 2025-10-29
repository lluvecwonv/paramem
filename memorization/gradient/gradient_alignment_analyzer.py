#!/usr/bin/env python3
"""
Gradient Alignment Analysis for Memorization Transfer Detection

Based on Definition 3.3:
Align(P(x), x; θ) = 1/K * Σ <g(x'_i; θ), g(x; θ)>

Computes cosine similarity between gradients of original text and paraphrases.
Uses influence_pile.py style gradient collection.
"""

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, PreTrainedModel
from tqdm import tqdm


class TextDataset(Dataset):
    """Simple text dataset for gradient collection"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}


def collect_gradients(model, tokenizer, texts, max_length=512):
    """
    Collect gradients for texts (influence_pile.py style)

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of texts
        max_length: Max sequence length

    Returns:
        grad_dict: {sample_id: {layer_name: gradient_tensor}}
    """
    dataset = TextDataset(texts, tokenizer, max_length)
    collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")
    loader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

    model.eval()
    grad_dict = {}

    for step, batch in enumerate(tqdm(loader, desc=f"Collecting gradients")):
        model.zero_grad()
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Next token prediction: shift labels
        batch['labels'] = batch['input_ids'][:, 1:].clone()  # 다음 토큰
        batch['input_ids'] = batch['input_ids'][:, :-1]      # 현재 토큰
        if 'attention_mask' in batch:
            batch['attention_mask'] = batch['attention_mask'][:, :-1]

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        sample_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Only track embedding and lm_head for efficiency (like influence_pile.py)
                if 'embed' in name or 'lm_head' in name:
                    sample_grads[name] = param.grad.cpu().clone().flatten()

        grad_dict[step] = sample_grads

        # Clean up to prevent memory issues
        del batch, outputs, loss
        torch.cuda.empty_cache()

    return grad_dict


def cosine_similarity(grad_dict1, grad_dict2):
    """
    Compute cosine similarity between two gradient dictionaries

    Args:
        grad_dict1: {layer_name: gradient_tensor}
        grad_dict2: {layer_name: gradient_tensor}

    Returns:
        float: Average cosine similarity across all layers
    """
    similarities = []

    for layer_name in grad_dict1.keys():
        if layer_name not in grad_dict2:
            continue

        grad1 = grad_dict1[layer_name]
        grad2 = grad_dict2[layer_name]

        # Cosine similarity = dot(g1, g2) / (||g1|| * ||g2||)
        dot_product = torch.sum(grad1 * grad2)
        norm1 = torch.norm(grad1)
        norm2 = torch.norm(grad2)

        cos_sim = dot_product / (norm1 * norm2 + 1e-12)
        similarities.append(cos_sim.item())

    # Average over all layers
    return np.mean(similarities) if similarities else 0.0


def compute_alignment_batch(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    original_texts: List[str],
    paraphrase_lists: List[List[str]],
    max_len: int = 512,
    device: str = "cuda",
    show_progress: bool = True
) -> List[float]:
    """
    Compute gradient alignments for multiple samples (influence_pile.py style).

    Args:
        model: Model to analyze
        tokenizer: Tokenizer
        original_texts: List of original texts
        paraphrase_lists: List of paraphrase lists (one list per original)
        max_len: Maximum sequence length
        device: Device to use (unused, model.device is used)
        show_progress: Show progress bar

    Returns:
        List of cosine similarity scores (one per original text)
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"Gradient Alignment Computation")
    print(f"{'='*60}")
    print(f"Original texts: {len(original_texts)}")
    print(f"Paraphrases per text: {len(paraphrase_lists[0]) if paraphrase_lists else 0}")
    print(f"{'='*60}\n")

    # Step 1: Collect gradients for original texts
    print("Step 1: Collecting gradients for original texts...")
    original_grad_dict = collect_gradients(model, tokenizer, original_texts, max_len)

    # Step 2: Collect gradients for each paraphrase set
    num_paraphrases = len(paraphrase_lists[0]) if paraphrase_lists else 0
    paraphrase_grad_dicts = []

    print(f"\nStep 2: Collecting gradients for {num_paraphrases} paraphrase sets...")
    for para_idx in range(num_paraphrases):
        print(f"  Paraphrase set {para_idx + 1}/{num_paraphrases}")
        # Extract para_idx-th paraphrase from each sample
        paraphrase_texts = [
            paras[para_idx] if para_idx < len(paras) else paras[0]
            for paras in paraphrase_lists
        ]
        para_grad_dict = collect_gradients(model, tokenizer, paraphrase_texts, max_len)
        paraphrase_grad_dicts.append(para_grad_dict)

    # Step 3: Compute cosine similarity
    print(f"\nStep 3: Computing gradient alignments...")
    alignment_scores = []

    for sample_id in tqdm(range(len(original_texts)), desc="Computing alignment"):
        orig_grads = original_grad_dict[sample_id]

        # Compute similarity with each paraphrase
        sample_similarities = []
        for para_grad_dict in paraphrase_grad_dicts:
            if sample_id not in para_grad_dict:
                continue

            para_grads = para_grad_dict[sample_id]
            similarity = cosine_similarity(orig_grads, para_grads)
            sample_similarities.append(similarity)

        # Average over all paraphrases
        avg_similarity = np.mean(sample_similarities) if sample_similarities else 0.0
        alignment_scores.append(avg_similarity)

    # Print summary
    mean_score = np.mean(alignment_scores)
    std_score = np.std(alignment_scores)

    print(f"\n{'='*60}")
    print(f"Mean Cosine Alignment: {mean_score:.4f} ± {std_score:.4f}")
    print(f"{'='*60}\n")

    return alignment_scores
