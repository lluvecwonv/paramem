"""
Utility functions for memorization analysis

This module provides minimal utilities for Pile paraphrase generation.
For full TOFU analysis utilities, see utils.py.backup
"""

import sys
import os
import logging
import datasets
import yaml

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Setup logger
logger = logging.getLogger(__name__)


def get_model_config(model_family):
    """Load model configuration from model_config.yaml

    Args:
        model_family (str): Model family name (e.g., 'pythia-2.8b', 'llama2-7b')

    Returns:
        dict: Model configuration containing hf_key, question_start_tag, etc.

    Raises:
        ValueError: If model_family is not found in config

    Example:
        >>> config = get_model_config('pythia-2.8b')
        >>> print(config['hf_key'])  # 'EleutherAI/pythia-2.8b'
    """
    config_path = os.path.join(os.path.dirname(__file__), '../config', 'model_config.yaml')

    with open(config_path, 'r') as f:
        model_configs = yaml.safe_load(f)

    if model_family not in model_configs:
        available_models = ', '.join(model_configs.keys())
        raise ValueError(f"Model family '{model_family}' not found in config. "
                        f"Available models: {available_models}")

    logger.info(f"Loaded config for {model_family}: {model_configs[model_family]['hf_key']}")
    return model_configs[model_family]


def load_pile_texts(data_path, subset_name='all', split='train', num_samples=None):
    """Load plain texts from The Pile dataset

    Simple utility function to load raw text from The Pile without tokenization.
    This is useful for paraphrase generation where we need the original text.

    Args:
        data_path (str): Dataset path (e.g., 'EleutherAI/pile') or local JSON path
        subset_name (str): Pile subset name (e.g., 'all', 'PubMed Abstracts')
        split (str): Dataset split (e.g., 'train', 'test')
        num_samples (int, optional): Number of samples to load (None for all)

    Returns:
        list: List of text strings

    Example:
        >>> texts = load_pile_texts('EleutherAI/pile', subset_name='all', split='train', num_samples=100)
        >>> print(f"Loaded {len(texts)} texts")
        >>> print(texts[0][:100])  # First 100 chars
    """
    logger.info(f"Loading Pile texts: {data_path}, subset={subset_name}, split={split}")

    if data_path.startswith('EleutherAI/pile'):
        data = datasets.load_dataset(data_path, subset_name, split=split, streaming=False, trust_remote_code=True)
    else:
        data = datasets.load_dataset('json', data_files=data_path, split=split, trust_remote_code=True)

    if num_samples and num_samples < len(data):
        data = data.select(range(num_samples))

    texts = [item['text'] for item in data]
    logger.info(f"✅ Loaded {len(texts)} texts from Pile")

    return texts
