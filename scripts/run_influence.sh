#!/bin/bash

# Influence Function Analysis for Pile Samples
# Usage: ./scripts/run_influence.sh [model] [domain] [hvp_method]
# Example: ./scripts/run_influence.sh pythia-2.8b arxiv gradient_match
# Example: ./scripts/run_influence.sh pythia-410m github DataInf

model=${1:-pythia-2.8b}
domain=${2:-arxiv}
hvp_method=${3:-gradient_match}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=========================================="
echo "Influence Function Analysis"
echo "=========================================="
echo "Model: $model"
echo "Domain: $domain"
echo "HVP Method: $hvp_method"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

cd /root/memorization/Influence-Functions

# Run influence analysis
python influence_pile.py \
    --model $model \
    --domain $domain \
    --data_dir ../pile_samples \
    --max_length 512 \
    --max_samples -1 \
    --hvp_method $hvp_method \
    --lambda_c 10 \
    --iter 3 \
    --alpha 1.0 \
    --grad_cache

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed!"
    echo "Results: cache/${model}_${domain}_${hvp_method}.csv"
else
    echo ""
    echo "❌ Analysis failed!"
    exit 1
fi
