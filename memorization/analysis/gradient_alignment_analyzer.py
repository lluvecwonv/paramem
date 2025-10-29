#!/usr/bin/env python3
"""
Gradient Alignment Analysis for Memorization Transfer Detection

Simple implementation based on Definition 3.3:
Align(P(x), x; θ) = 1/K * Σ <g(x'_i; θ), g(x; θ)>

ALIGNMENT METRICS EXPLANATION:
1. Inner Product (내적): <g(x), g(x')>
   - 원시 그라디언트 벡터 간의 내적
   - 크기(magnitude)에 민감: 큰 그라디언트일수록 높은 값
   - 모델, 배치, 파라미터 선택에 따라 스케일이 크게 달라짐
   - 절대적 비교가 어려우므로 보조 지표로 사용 권장

2. Cosine Similarity (코사인 유사도): <g(x), g(x')> / (||g(x)|| * ||g(x')||)
   - 정규화된 그라디언트 간의 각도 유사성 (0~1)
   - 크기에 무관하게 방향성만 측정
   - 서로 다른 모델/설정 간 비교가 용이
   - 해석이 직관적: 1에 가까울수록 강한 정렬, 0에 가까울수록 약한 정렬
   - 메인 지표로 사용 권장

USAGE RECOMMENDATION:
- 보고서나 모델 비교에는 cosine similarity를 주 지표로 사용
- inner product는 gradient magnitude 분석 시에만 참고
- 암기 전이 강도 평가: cosine > 0.7 (강함), 0.3~0.7 (보통), < 0.3 (약함)
"""

import torch
from torch import nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import logging

logger = logging.getLogger(__name__)


def iter_trainable_params(model: PreTrainedModel, only_last_n_layers: Optional[int] = None) -> List[nn.Parameter]:
    """필요 시 마지막 N개 레이어만 그라디언트 계산(가속/메모리 절약)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if only_last_n_layers is None:
        return params
    # 간단하게 끝에서 N개의 파라미터 텐서를 사용 (레이어 단위로 더 정교화 가능)
    return params[-only_last_n_layers:]


def _grads_to_cpu_fp32(params: List[nn.Parameter]) -> List[torch.Tensor]:
    """현재 params의 grad를 CPU FP32 텐서(평탄) 리스트로 반환."""
    cpu_grads: List[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            cpu_grads.append(torch.empty(0, dtype=torch.float32))
            continue
        g_cpu = p.grad.detach().to(dtype=torch.float32, device="cpu").view(-1).contiguous()
        cpu_grads.append(g_cpu)
    return cpu_grads


def _cpu_grads_norm(cpu_grads: List[torch.Tensor]) -> float:
    norm_sq = 0.0
    for g in cpu_grads:
        if g.numel() == 0:
            continue
        g_fp32 = g.to(dtype=torch.float32)
        norm_sq += float(torch.dot(g_fp32, g_fp32))
    return float(norm_sq ** 0.5 + 1e-12)


def _dot_with_cpu_grads(params: List[nn.Parameter], cpu_grads_ref: List[torch.Tensor]) -> Tuple[float, float]:
    """현재 params.grad와 CPU에 저장된 기준 grad 간 내적과 현재 grad 노름을 계산.
    - 내적과 노름은 CPU에서 계산해 GPU 메모리를 최소화한다.
    """
    dot_sum = 0.0
    norm_sq = 0.0
    for p, g_ref in zip(params, cpu_grads_ref):
        if p.grad is None or g_ref.numel() == 0:
            continue
        g_cpu = p.grad.detach().to(dtype=torch.float32, device="cpu").view(-1)
        g_ref_fp32 = g_ref.to(dtype=torch.float32)
        dot_sum += float(torch.dot(g_cpu, g_ref_fp32))
        norm_sq += float(torch.dot(g_cpu, g_cpu))
    return dot_sum, float(norm_sq ** 0.5 + 1e-12)


def sequence_loss(model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    labels는 causal LM 관례대로 input_ids를 한 칸 시프트해 계산.
    padding 토큰은 -100으로 무시.
    """
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # HF는 평균 토큰 CE를 반환(무시 토큰 제외) → 정의 3.1의 시퀀스 평균 손실과 합치게 사용
    return out.loss


def sequence_backward(
    model: nn.Module,
    params: List[nn.Parameter],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    fp16_autocast: bool = False,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    """
    시퀀스에 대한 backward pass 수행
    
    Args:
        grad_scaler: FP16 사용 시 수치적 안정성을 위한 gradient scaler.
                    일반적으로 inference+backward에서는 필요하지 않지만,
                    매우 작은 모델이나 배치에서 underflow 위험이 있을 수 있음.
    """
    model.zero_grad(set_to_none=True)
    # NOTE: 기존 full-backward 경로는 비권장(성능 비효율). 유지만 하고 사용은 지양.
    # 디바이스에 따른 안전한 autocast 처리
    if input_ids.is_cuda:
        amp_ctx = torch.autocast("cuda", dtype=torch.float16, enabled=fp16_autocast)
    else:
        amp_ctx = nullcontext()
    with amp_ctx:
        loss = sequence_loss(model, input_ids, attention_mask)
    # 일반적인 backward (분석 목적에선 GradScaler 불사용 권장)
    loss.backward()
    
    # grads are now stored in params
    return None


@dataclass
class AlignOutputs:
    align_inner: float
    align_cosine: float
    grad_norm_orig: float
    grad_norm_para_mean: float


# ------------------------------
# Head-only gradient alignment
# ------------------------------

def _final_norm(model: PreTrainedModel, hidden: torch.Tensor) -> torch.Tensor:
    """모델 유형에 따라 최종 LayerNorm을 적용(GPT: ln_f, LLaMA: model.norm)."""
    # LLaMA 계열
    if hasattr(getattr(model, "model", None), "norm"):
        return model.model.norm(hidden)
    # GPT2/NeoX 계열
    if hasattr(getattr(model, "transformer", None), "ln_f"):
        return model.transformer.ln_f(hidden)
    return hidden


def _encode(tokenizer: AutoTokenizer, texts, device: str, max_len: int):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def _backbone_hidden(model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """백본은 no_grad로만 사용, 마지막 히든을 추출 후 필요 시 최종 Norm 적용."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hidden_last = out.hidden_states[-1]
    return _final_norm(model, hidden_last)


def _masked_ce(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Causal LM 시프트 CE를 마스크 평균으로 계산(HF의 유효 토큰 평균과 일치)."""
    tgt = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].contiguous()
    logits_use = logits[:, :-1, :].contiguous()
    v = logits_use.size(-1)
    loss_tok = F.cross_entropy(logits_use.view(-1, v), tgt.view(-1), reduction="none")
    loss_tok = loss_tok.view_as(tgt)
    denom = mask.sum().clamp_min(1)
    return (loss_tok * mask).sum() / denom


def _grad_head_vector(
    model: PreTrainedModel,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_autocast: bool,
) -> Tuple[torch.Tensor, float]:
    """출력 헤드 파라미터에 대해서만 그라디언트 벡터를 계산하고 CPU FP32로 반환."""
    head = model.get_output_embeddings()
    if head is None:
        # fallback (일부 모델은 lm_head만 존재)
        head = getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError("Output head (lm_head) module을 찾을 수 없습니다.")

    params = [p for p in head.parameters()]
    # 안전한 autocast (CUDA에서만)
    amp_ctx = torch.autocast("cuda", dtype=torch.float16, enabled=use_autocast) if hidden.is_cuda else nullcontext()
    with amp_ctx:
        logits = head(hidden)  # [B, T, V]
        loss = _masked_ce(logits, input_ids, attention_mask)

    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    flat = []
    for g in grads:
        if g is None:
            continue
        flat.append(g.detach().to("cpu", dtype=torch.float32).view(-1))
    g_flat = torch.cat(flat, dim=0) if flat else torch.zeros(0, dtype=torch.float32)
    norm = float(torch.linalg.vector_norm(g_flat).item() + 1e-12)
    return g_flat, norm


def paraphrase_alignment_for_one(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    orig_text: str,
    paraphrases: List[str],
    params: Optional[List[nn.Parameter]] = None,  # deprecated in head-only path
    max_len: int = 512,
    device: str = "cuda",
    use_cosine: bool = True,
    fp16_autocast: bool = False,
) -> AlignOutputs:
    """헤드(출력층) 파라미터에 대해서만 그라디언트 정렬을 계산(효율 모드)."""
    # 원본 히든 (백본은 no_grad)
    input_ids, attn = _encode(tokenizer, [orig_text], device, max_len)
    hidden = _backbone_hidden(model, input_ids, attn)
    g_ref, ref_norm = _grad_head_vector(model, hidden, input_ids, attn, fp16_autocast)

    inner_sum = 0.0
    cosine_sum = 0.0
    para_norm_sum = 0.0
    for p_text in paraphrases:
        input_ids_p, attn_p = _encode(tokenizer, [p_text], device, max_len)
        hidden_p = _backbone_hidden(model, input_ids_p, attn_p)
        g_p, p_norm = _grad_head_vector(model, hidden_p, input_ids_p, attn_p, fp16_autocast)

        # 내적/코사인 계산 (CPU FP32)
        # 길이 불일치 가능성은 낮지만 안전하게 공통 길이로 처리
        min_len = min(g_ref.numel(), g_p.numel())
        dot_val = float(torch.dot(g_ref[:min_len], g_p[:min_len]).item()) if min_len > 0 else 0.0
        inner_sum += dot_val
        para_norm_sum += p_norm
        if use_cosine:
            cosine_sum += dot_val / (ref_norm * p_norm + 1e-12)

    n = max(1, len(paraphrases))
    align_inner = inner_sum / n
    align_cosine = (cosine_sum / n) if use_cosine else float("nan")
    return AlignOutputs(
        align_inner=align_inner,
        align_cosine=align_cosine,
        grad_norm_orig=float(ref_norm),
        grad_norm_para_mean=float(para_norm_sum / n),
    )


class GradientAlignmentAnalyzer:
    """Gradient Alignment 분석기"""
    
    def __init__(self, 
                 use_cosine: bool = True,
                 fp16_autocast: bool = False,
                 only_last_n_layers: Optional[int] = None,
                 max_len: int = 512):
        self.use_cosine = use_cosine
        self.fp16_autocast = fp16_autocast
        self.only_last_n_layers = only_last_n_layers
        self.max_len = max_len
        
    def analyze_batch(self,
                     model: nn.Module,
                     tokenizer: AutoTokenizer,
                     original_texts: List[str],
                     paraphrase_texts_list: List[List[str]],
                     device: str = "cuda") -> Dict:
        """
        배치 처리로 여러 원문과 패러프레이즈의 gradient alignment 계산
        
        Args:
            model: 분석할 모델
            tokenizer: 토크나이저
            original_texts: 원본 텍스트 리스트
            paraphrase_texts_list: 각 원본에 대한 패러프레이즈 리스트들
            device: 디바이스
            
        Returns:
            분석 결과 딕셔너리
        """
        results = {
            'alignments': [],
            'align_inner_scores': [],
            'align_cosine_scores': [],
            'gradient_norms': {
                'original': [],
                'paraphrase': []
            }
        }
        
        # head-only 경로에서는 params 서브셋이 필요 없음
        model.eval()  # 일관성을 위해 eval 모드
        
        logger.info(f"Computing gradient alignments for {len(original_texts)} samples...")
        
        for i, (orig_text, paraphrases) in enumerate(zip(original_texts, paraphrase_texts_list)):
            if not paraphrases:  # 패러프레이즈가 없으면 스킵
                continue
                
            try:
                # 개별 alignment 계산
                align_result = paraphrase_alignment_for_one(
                    model=model,
                    tokenizer=tokenizer,
                    orig_text=orig_text,
                    paraphrases=paraphrases,
                    max_len=self.max_len,
                    device=device,
                    use_cosine=self.use_cosine,
                    fp16_autocast=self.fp16_autocast
                )
                
                # 결과 저장
                results['alignments'].append(align_result)
                results['align_inner_scores'].append(align_result.align_inner)
                results['align_cosine_scores'].append(align_result.align_cosine)
                
                # 그라디언트 노름 저장 (스칼라 기반)
                results['gradient_norms']['original'].append(align_result.grad_norm_orig)
                results['gradient_norms']['paraphrase'].append(align_result.grad_norm_para_mean)
                
                if (i + 1) % 5 == 0:
                    logger.debug(f"Processed {i + 1}/{len(original_texts)} samples")
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        # 통계 계산
        results['statistics'] = self._compute_statistics(results)
        
        return results
    
    def _compute_statistics(self, results: Dict) -> Dict:
        """결과 통계 계산"""
        inner_scores = [x for x in results['align_inner_scores'] if not torch.isnan(torch.tensor(x))]
        cosine_scores = [x for x in results['align_cosine_scores'] if not torch.isnan(torch.tensor(x))]
        
        stats = {
            'num_samples': len(results['alignments']),
            'mean_align_inner': sum(inner_scores) / max(len(inner_scores), 1),
            'std_align_inner': torch.tensor(inner_scores).std().item() if len(inner_scores) > 1 else 0.0,
            'mean_align_cosine': sum(cosine_scores) / max(len(cosine_scores), 1),
            'std_align_cosine': torch.tensor(cosine_scores).std().item() if len(cosine_scores) > 1 else 0.0,
            'mean_grad_norm_orig': sum(results['gradient_norms']['original']) / max(len(results['gradient_norms']['original']), 1),
            'mean_grad_norm_para': sum(results['gradient_norms']['paraphrase']) / max(len(results['gradient_norms']['paraphrase']), 1)
        }
        
        return stats
    
    def compare_models(self,
                      model1: nn.Module,
                      model2: nn.Module, 
                      tokenizer: AutoTokenizer,
                      original_texts: List[str],
                      paraphrase_texts_list: List[List[str]],
                      device: str = "cuda",
                      model1_name: str = "Model1",
                      model2_name: str = "Model2") -> Dict:
        """두 모델 간의 gradient alignment 비교"""
        
        logger.info(f"Comparing gradient alignments between {model1_name} and {model2_name}")
        
        # 각 모델의 alignment 계산
        results1 = self.analyze_batch(model1, tokenizer, original_texts, paraphrase_texts_list, device)
        results2 = self.analyze_batch(model2, tokenizer, original_texts, paraphrase_texts_list, device)
        
        # 비교 결과
        comparison = {
            model1_name: results1,
            model2_name: results2,
            'comparison': {
                'inner_alignment_diff': results2['statistics']['mean_align_inner'] - results1['statistics']['mean_align_inner'],
                'cosine_alignment_diff': results2['statistics']['mean_align_cosine'] - results1['statistics']['mean_align_cosine'],
                'grad_norm_orig_diff': results2['statistics']['mean_grad_norm_orig'] - results1['statistics']['mean_grad_norm_orig'],
                'grad_norm_para_diff': results2['statistics']['mean_grad_norm_para'] - results1['statistics']['mean_grad_norm_para']
            }
        }
        
        return comparison


def create_alignment_report(analysis_results: Dict, title: str = "Gradient Alignment Analysis") -> str:
    """분석 결과 리포트 생성"""
    report = []
    report.append("=" * 60)
    report.append(title)
    report.append("=" * 60)
    
    if 'statistics' in analysis_results:
        stats = analysis_results['statistics']
        report.append(f"Samples processed: {stats['num_samples']}")
        report.append(f"Mean Inner Alignment: {stats['mean_align_inner']:.4f} (±{stats['std_align_inner']:.4f})")
        report.append(f"Mean Cosine Alignment: {stats['mean_align_cosine']:.4f} (±{stats['std_align_cosine']:.4f})")
        report.append(f"Mean Gradient Norm (Original): {stats['mean_grad_norm_orig']:.4f}")
        report.append(f"Mean Gradient Norm (Paraphrase): {stats['mean_grad_norm_para']:.4f}")
        report.append("")
        
        # 해석
        cosine_align = stats['mean_align_cosine']
        if cosine_align > 0.7:
            report.append("🔥 High gradient alignment - Strong memorization transfer detected")
        elif cosine_align > 0.3:
            report.append("⚠️  Moderate gradient alignment - Partial memorization transfer") 
        else:
            report.append("✅ Low gradient alignment - Weak memorization transfer")
    
    report.append("=" * 60)
    return "\n".join(report)
