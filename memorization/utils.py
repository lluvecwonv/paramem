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


def get_model_identifiers_from_yaml(model_family):
    """Alias for get_model_config for backward compatibility"""
    return get_model_config(model_family)


def find_train_test_pairs(results_dir, domains=None):
    """Find train/test file pairs in results directory

    Args:
        results_dir (str): Directory containing result JSON files
        domains (list, optional): List of domains to filter (e.g., ['arxiv', 'dm_mathematics'])
                                 If None, returns all pairs

    Returns:
        dict: Dictionary mapping base_name to {'train': path, 'test': path, 'domain': domain}

    Example:
        >>> pairs = find_train_test_pairs('./results')
        >>> for name, files in pairs.items():
        ...     print(f"{name}: {files['train']} & {files['test']}")

        >>> # Filter by domains
        >>> pairs = find_train_test_pairs('./results', domains=['arxiv', 'dm_mathematics'])
    """
    import re
    from pathlib import Path
    from collections import defaultdict

    # Default domains from MIA analysis
    if domains is None:
        domains = [
            "arxiv",
            "dm_mathematics",
            "github",
            "hackernews",
            "pile_cc",
            "pubmed_central",
            "wikipedia_en"
        ]

    results_path = Path(results_dir)
    json_files = list(results_path.glob("*.json"))

    pairs = defaultdict(dict)
    pattern = r"^(.+)_(train|test)\.json$"

    for json_file in json_files:
        match = re.match(pattern, json_file.name)
        if match:
            base_name = match.group(1)
            split_type = match.group(2)

            # Extract domain from base_name
            # Pattern: {model}_{method}_{domain}
            domain = None
            for d in domains:
                if d in base_name:
                    domain = d
                    break

            # Skip if domain not in filter list
            if domain is None:
                continue

            if base_name not in pairs:
                pairs[base_name]['domain'] = domain

            pairs[base_name][split_type] = str(json_file)

    # Return only complete pairs (both train and test)
    complete_pairs = {
        name: files for name, files in pairs.items()
        if 'train' in files and 'test' in files
    }

    logger.info(f"Found {len(complete_pairs)} train/test pairs in {results_dir}")
    if domains:
        logger.info(f"Filtered by domains: {domains}")

    return complete_pairs
