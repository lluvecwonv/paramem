#!/bin/bash

# Gradient Alignment Analysis Script for Pile/Memorization
# Usage: ./run_gradient_alignment.sh [model_family] [paraphrase_results_file]
# Example: ./run_gradient_alignment.sh pythia-2.8b results/pythia-2.8b_greedy_Nprompts_train_pile/generated_paraphrases.json

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# GPU 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Get parameters
model_family=${1:-pythia-2.8b}
paraphrase_results_file=${2}

echo "========================================="
echo "Gradient Alignment Analysis"
echo "========================================="
echo "Model family: $model_family"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Check if paraphrase results file is provided
if [ -z "$paraphrase_results_file" ]; then
    echo "❌ Error: Paraphrase results file not specified!"
    echo ""
    echo "Usage: ./run_gradient_alignment.sh [model_family] [paraphrase_results_file]"
    echo ""
    echo "Example:"
    echo "  ./run_gradient_alignment.sh pythia-2.8b results/pythia-2.8b_greedy_Nprompts_train_pile/generated_paraphrases.json"
    exit 1
fi

# Check if file exists
if [ ! -f "$paraphrase_results_file" ]; then
    echo "❌ Error: File not found: $paraphrase_results_file"
    exit 1
fi

echo "Paraphrase results: $paraphrase_results_file"
echo ""

# Auto-detect model path
case $model_family in
    pythia-2.8b)
        model_path="EleutherAI/pythia-2.8b"
        ;;
    pythia-1.4b)
        model_path="EleutherAI/pythia-1.4b"
        ;;
    pythia-410m)
        model_path="EleutherAI/pythia-410m"
        ;;
    llama2-7b)
        model_path="NousResearch/Llama-2-7b-hf"
        ;;
    phi)
        model_path="microsoft/phi-2"
        ;;
    *)
        echo "❌ Error: Unknown model family: $model_family"
        exit 1
        ;;
esac

echo "✅ Model path: $model_path"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Create output filename
output_dir=$(dirname "$paraphrase_results_file")
output_file="${output_dir}/gradient_alignment_results.json"

echo "🚀 Running Gradient Alignment Analysis..."
echo "Output: $output_file"
echo ""

# Run analysis
python memorization/gradient_alignment_main.py \
    model_family=$model_family \
    model_path="$model_path" \
    paraphrase_results_file="$paraphrase_results_file" \
    output_file="$output_file" \
    max_length=512

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    echo "Results: $output_file"
else
    echo ""
    echo "❌ Analysis failed!"
    exit 1
fi
