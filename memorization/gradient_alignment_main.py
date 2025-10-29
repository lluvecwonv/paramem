#!/usr/bin/env python3

import os
import sys
import json
import torch
import hydra
import logging
import warnings
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# 현재 스크립트 디렉토리 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from utils import get_model_identifiers_from_yaml
from memorization.gradient.gradient_alignment_analyzer import compute_alignment_batch

# 경고 억제
warnings.filterwarnings("ignore", category=UserWarning)

# CUDA 메모리 최적화
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_paraphrase_results(json_path):
    """패러프레이즈 결과 JSON 파일 로드"""
    logger.info(f"Loading paraphrase results from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_texts = []
    paraphrase_texts_list = []
    original_memorization_scores = []

    for item in data.get('results', []):
        original_question = item.get('Question', '')
        if not original_question:
            continue

        original_texts.append(original_question)
        original_memorization_scores.append(item.get('OriginalMemorization'))

        # 패러프레이즈 수집
        paraphrases = []
        for para_result in item.get('ParaphraseResults', []):
            paraphrased_question = para_result.get('paraphrased_question', '')
            if paraphrased_question and paraphrased_question != original_question:
                paraphrases.append(paraphrased_question)

        # 패러프레이즈가 없으면 원본 사용
        paraphrase_texts_list.append(paraphrases if paraphrases else [original_question])

    logger.info(f"✅ Loaded {len(original_texts)} questions, {sum(len(p) for p in paraphrase_texts_list)} total paraphrases")
    return original_texts, paraphrase_texts_list, original_memorization_scores


def print_analysis_report(scores):
    """분석 결과 출력"""
    mean_score = np.mean(scores) if scores else 0.0
    std_score = np.std(scores) if scores else 0.0

    logger.info("=" * 60)
    logger.info("📊 GRADIENT ALIGNMENT ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total processed samples: {len(scores)}")
    logger.info(f"Mean Cosine Alignment: {mean_score:.6f} ± {std_score:.6f}")

    # 암기 전이 평가
    if mean_score > 0.7:
        logger.info("🔥 High gradient alignment - Strong memorization transfer")
    elif mean_score > 0.3:
        logger.info("⚠️  Moderate gradient alignment - Partial memorization transfer")
    else:
        logger.info("✅ Low gradient alignment - Weak memorization transfer")

    # 분포 분석
    high_align = sum(1 for score in scores if score > 0.7)
    med_align = sum(1 for score in scores if 0.3 <= score <= 0.7)
    low_align = sum(1 for score in scores if score < 0.3)
    total = len(scores)

    if total > 0:
        logger.info(f"Distribution: High({high_align/total*100:.1f}%) | Med({med_align/total*100:.1f}%) | Low({low_align/total*100:.1f}%)")
    logger.info("=" * 60)


def save_results(output_file, cfg, scores, original_texts, paraphrase_texts_list, paraphrase_json_path):
    """결과를 JSON 파일로 저장"""
    logger.info("💾 Saving results...")

    mean_score = np.mean(scores) if scores else 0.0
    std_score = np.std(scores) if scores else 0.0

    # 분포 분석
    high_align = sum(1 for score in scores if score > 0.7)
    med_align = sum(1 for score in scores if 0.3 <= score <= 0.7)
    low_align = sum(1 for score in scores if score < 0.3)
    total = len(scores) if scores else 1

    results_dict = {
        'config': {
            'model_family': cfg.model_family,
            'model_path': cfg.model_path,
            'total_questions': len(original_texts),
            'total_samples': len(scores),
            'analysis_type': 'gradient_alignment',
            'paraphrase_source': paraphrase_json_path
        },
        'overall_statistics': {
            'num_samples': len(scores),
            'mean_cosine_alignment': float(mean_score),
            'std_cosine_alignment': float(std_score),
        },
        'individual_results': [
            {
                'sample_index': i,
                'original_text': orig[:100] + "..." if len(orig) > 100 else orig,
                'num_paraphrases': len(paras),
                'cosine_alignment': float(score)
            }
            for i, (orig, paras, score) in enumerate(zip(original_texts, paraphrase_texts_list, scores))
        ],
        'distribution_analysis': {
            'high_alignment_count': high_align,
            'medium_alignment_count': med_align,
            'low_alignment_count': low_align,
            'high_alignment_percent': high_align/total*100,
            'medium_alignment_percent': med_align/total*100,
            'low_alignment_percent': low_align/total*100
        },
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Results saved to: {output_file}")


@hydra.main(version_base=None, config_path="config", config_name="gradient_alignment")
def main(cfg: DictConfig):
    """그라디언트 정렬도 분석 메인 함수"""

    # Local rank 설정 (분산 학습 지원)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    logger.info("🔥 Starting Gradient Alignment Analysis for TOFU")
    logger.info(f"🔧 Model family: {cfg.model_family}")
    logger.info(f"🔧 Model path: {cfg.model_path}")

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
        logger.info(f"🔧 Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        if world_size > 1:
            logger.info(f"🔧 Distributed mode: Rank {local_rank}/{world_size}")
    else:
        device = torch.device('cpu')
        logger.info("🔧 Using CPU")

    try:
        # 모델과 토크나이저 로드
        logger.info("📥 Loading model and tokenizer...")
        model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
        model_id = model_cfg["hf_key"]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model from: {cfg.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        model.eval()
        logger.info("✅ Model loaded successfully")

        # 패러프레이즈 결과 로드
        paraphrase_json_path = cfg.paraphrase_results_file
        if not os.path.exists(paraphrase_json_path):
            raise FileNotFoundError(f"Paraphrase results file not found: {paraphrase_json_path}")

        original_texts, paraphrase_texts_list, original_memorization_scores = \
            load_paraphrase_results(paraphrase_json_path)

        # Gradient Alignment 계산
        logger.info("🚀 Computing gradient alignments...")

        scores = compute_alignment_batch(
            model=model,
            tokenizer=tokenizer,
            original_texts=original_texts,
            paraphrase_lists=paraphrase_texts_list,
            max_len=cfg.max_length,
            device=str(device)
        )

        logger.info(f"✅ Successfully processed {len(scores)} samples")

        # 결과 출력
        print_analysis_report(scores)

        # 결과 저장
        save_results(
            output_file=cfg.output_file,
            cfg=cfg,
            scores=scores,
            original_texts=original_texts,
            paraphrase_texts_list=paraphrase_texts_list,
            paraphrase_json_path=paraphrase_json_path
        )

        logger.info("🎯 Gradient Alignment Analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
