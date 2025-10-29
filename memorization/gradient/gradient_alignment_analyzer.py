#!/usr/bin/env python3
"""
Gradient Alignment Analysis for Memorization Transfer Detection

Based on Definition 3.3:
Align(P(x), x; θ) = 1/K * Σ <g(x'_i; θ), g(x; θ)>

Computes cosine similarity between gradients of original text and paraphrases.
"""

import torch
from torch import nn
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, PreTrainedModel
from tqdm import tqdm


def _grads_to_cpu(params: List[nn.Parameter]) -> List[torch.Tensor]:
    """Extract gradients to CPU as flattened FP32 tensors."""
    cpu_grads = []
    for p in params:
        if p.grad is None:
            cpu_grads.append(torch.empty(0, dtype=torch.float32))
        else:
            cpu_grads.append(p.grad.detach().cpu().float().view(-1).contiguous())
    return cpu_grads


def _compute_norm(grads: List[torch.Tensor]) -> float:
    """Compute L2 norm of gradient list."""
    norm_sq = sum(torch.dot(g, g).item() for g in grads if g.numel() > 0)
    return (norm_sq ** 0.5) + 1e-12


def _compute_dot(params: List[nn.Parameter], ref_grads: List[torch.Tensor]) -> Tuple[float, float]:
    """Compute dot product and norm between current gradients and reference gradients."""
    dot_sum = 0.0
    norm_sq = 0.0

    for p, g_ref in zip(params, ref_grads):
        if p.grad is None or g_ref.numel() == 0:
            continue
        g_curr = p.grad.detach().cpu().float().view(-1)
        dot_sum += torch.dot(g_curr, g_ref).item()
        norm_sq += torch.dot(g_curr, g_curr).item()

    return dot_sum, (norm_sq ** 0.5) + 1e-12


def compute_loss_and_backward(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> None:
    """Compute loss and backward pass."""
    model.zero_grad(set_to_none=True)

    # Prepare labels (shift for causal LM)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Forward + backward
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()


def compute_alignment(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    orig_text: str,
    paraphrases: List[str],
    params: Optional[List[nn.Parameter]] = None,
    max_len: int = 512,
    device: str = "cuda"
) -> float:
    """
    Compute gradient alignment between original text and paraphrases.

    Returns:
        Average cosine similarity between gradients
    """
    # Get parameters to compute gradients for
    if params is None:
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            params = list(model.transformer.h[-1].parameters())
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            params = list(model.model.layers[-1].parameters())
        else:
            params = [p for p in model.parameters() if p.requires_grad]

    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(text: str):
        enc = tokenizer(text, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len)
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # Compute gradient for original text
    input_ids, attn = encode(orig_text)
    compute_loss_and_backward(model, input_ids,  attn)
    orig_grads = _grads_to_cpu(params)
    orig_norm = _compute_norm(orig_grads)

    # Compute alignment with each paraphrase
    cosine_sum = 0.0
    for para_text in paraphrases:
        input_ids, attn = encode(para_text)
        compute_loss_and_backward(model, input_ids, attn)

        dot_val, para_norm = _compute_dot(params, orig_grads)
        cosine_sum += dot_val / (orig_norm * para_norm)

        torch.cuda.empty_cache()

    return cosine_sum / max(1, len(paraphrases))


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
    Compute gradient alignments for multiple samples.

    Args:
        model: Model to analyze
        tokenizer: Tokenizer
        original_texts: List of original texts
        paraphrase_lists: List of paraphrase lists (one list per original)
        max_len: Maximum sequence length
        device: Device to use
        show_progress: Show progress bar

    Returns:
        List of cosine similarity scores
    """
    model.eval()

    # Get parameters once
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        params = list(model.transformer.h[-1].parameters())
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        params = list(model.model.layers[-1].parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    scores = []
    iterator = zip(original_texts, paraphrase_lists)

    if show_progress:
        total_paraphrases = sum(len(p) for p in paraphrase_lists)
        iterator = tqdm(iterator, total=len(original_texts),
                       desc="Computing alignments",
                       unit="sample",
                       postfix={'avg_score': 0.0})

    for i, (orig, paras) in enumerate(iterator):
        if not paras:
            continue
        score = compute_alignment(model, tokenizer, orig, paras, params, max_len, device)
        scores.append(score)

        # Update progress bar with current average
        if show_progress and scores:
            avg_score = sum(scores) / len(scores)
            iterator.set_postfix({
                'avg_score': f'{avg_score:.4f}',
                'last_score': f'{score:.4f}',
                'paras': len(paras)
            })

        # Periodic memory cleanup
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    return scores
