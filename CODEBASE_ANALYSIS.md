# 📊 /root/memorization 코드베이스 분석

Generated: 2025-10-28

================================================================================
## 🏗️ 전체 구조 개요
================================================================================

```
/root/memorization/
└── mimir/                          # MIMIR 패키지 (LLM Memorization 측정 도구)
    ├── mimir/                      # Core MIMIR 라이브러리
    │   └── attacks/               # Membership Inference Attacks 구현
    ├── memorization/              # 🎯 커스텀 Memorization 분석 모듈 (TOFU 연동)
    │   ├── analysis/              # Paraphrase 분석 도구
    │   ├── config/                # 설정 파일들
    │   ├── gradient/              # Gradient alignment 분석
    │   ├── prompts/               # Paraphrase 생성 프롬프트
    │   └── utils.py               # 유틸리티 함수들
    ├── data/                      # 데이터 파일
    ├── configs/                   # MIMIR 실험 설정
    ├── scripts/                   # 실행 스크립트
    ├── analysis/                  # 결과 분석 스크립트
    ├── paraphrase_main.py         # 🔥 Paraphrase 분석 메인 (TOFU 연동)
    └── run.py                     # MIMIR MIA 실험 메인
```

================================================================================
## 📦 1. MIMIR 패키지 (Core Library)
================================================================================

### 목적
- LLM의 Memorization을 측정하기 위한 패키지
- Membership Inference Attack (MIA) 구현
- 논문: "Do Membership Inference Attacks Work on Large Language Models?"

### 주요 컴포넌트

#### 1.1 `mimir/attacks/` - MIA 공격 구현
```
attacks/
├── loss.py              # Likelihood-based attack
├── ref.py               # Reference-based attack
├── zlib.py              # Zlib entropy attack
├── neighborhood.py      # Neighborhood attack
├── min_k.py             # Min-K% Prob attack
├── min_k_plus_plus.py   # Min-K%++ attack
├── gradnorm.py          # Gradient Norm attack
├── recall.py            # ReCaLL attack
├── dc_pdd.py            # DC-PDD attack
└── all_attacks.py       # Attack 인터페이스
```

**각 공격 방법 설명:**

1. **Likelihood (loss)**: 단순히 타겟 데이터의 likelihood를 스코어로 사용
2. **Reference-based (ref)**: Reference 모델의 스코어로 정규화
3. **Zlib Entropy**: Zlib 압축 크기로 샘플 난이도 추정
4. **Neighborhood (ne)**: 보조 모델로 이웃 생성, likelihood 변화 측정
5. **Min-K% Prob (min_k)**: 최소 likelihood를 가진 k% 토큰 사용
6. **Min-K%++ (min_k++)**: 정규화된 likelihood로 Min-K% 개선
7. **Gradient Norm (gradnorm)**: 타겟 데이터의 gradient norm 사용
8. **ReCaLL**: Unconditional vs conditional log-likelihood 비교
9. **DC-PDD**: 대규모 말뭉치의 빈도 분포로 토큰 확률 보정

#### 1.2 `run.py` - MIMIR 실험 실행
```python
# 사용법
python run.py --config configs/mi.json
```

- MIA 실험 실행
- 다양한 공격 방법 테스트
- 결과를 `results/` 디렉토리에 저장

================================================================================
## 🎯 2. memorization/ 모듈 (커스텀 TOFU 연동)
================================================================================

### 목적
- TOFU unlearning 데이터셋과 연동
- Paraphrase 기반 memorization 측정
- Gradient alignment 분석

### 디렉토리 구조

```
memorization/
├── analysis/                    # Paraphrase 분석 도구
│   ├── paraphrase_analyzer.py  # Dual model 분석기
│   ├── paraphrase_generator.py # Paraphrase 생성기
│   ├── paraphrase_visualizer.py# 시각화 도구
│   └── paraphrase_generation_utils.py  # 생성 유틸
│
├── config/                      # 설정 파일
│   ├── model_config.yaml       # 모델별 설정 (llama2-7b, phi 등)
│   ├── paraphrase_analysis.yaml# Paraphrase 분석 설정
│   └── gradient_alignment.yaml # Gradient alignment 설정
│
├── gradient/                    # Gradient 분석
│   ├── gradient_calculator.py  # Gradient 계산
│   └── alignment_analyzer.py   # Alignment 분석
│
├── prompts/                     # Paraphrase 프롬프트
│   ├── near_duplicate_1.txt    # 2단어 변경 제약
│   ├── near_duplicate_2.txt
│   ├── near_duplicate_3.txt
│   ├── near_duplicate_4.txt
│   └── near_duplicate_5.txt
│
├── gradient_alignment_main.py  # Gradient alignment 실험
├── utils.py                    # 유틸리티 함수들
└── notebook.ipynb              # 분석 노트북
```

================================================================================
## 🔥 3. paraphrase_main.py (TOFU 연동 메인)
================================================================================

### 역할
- TOFU 데이터셋에서 paraphrase 기반 memorization 측정
- Full model vs Retain model 비교 분석

### Workflow

```
Step 1: Load Models
  ├── Full model (trained on all data)
  └── Retain model (trained without forget set)

Step 2: Analyze Original Questions
  ├── Generate answers with both models
  ├── Compute memorization scores (acc_in - acc_out)
  └── Save to original_*.json

Step 3: Generate Paraphrases
  ├── Use ParaphraseGenerator
  ├── Apply generation mode (beam_1prompt, beam_5prompts, etc.)
  └── Generate N paraphrases per question

Step 4: Analyze Paraphrased Questions
  ├── Generate answers with both models
  ├── Compute memorization scores
  └── Save to paraphrase_*.json

Step 5: Compare & Visualize
  ├── Compare original vs paraphrase scores
  ├── Generate plots
  └── Save results to results/{model}_{mode}_{split}/
```

### 실행 예시

```bash
# TOFU 프로젝트에서 실행
cd /root/memorization_unlearn/TOFU
./run_gradient_alignment.sh
```

================================================================================
## 📊 4. 주요 클래스 및 함수
================================================================================

### 4.1 ParaphraseGenerator (analysis/paraphrase_generator.py)

**목적**: 다양한 전략으로 paraphrase 생성

**Generation Modes**:
1. `greedy_5prompts`: Greedy decoding + 5 diverse prompts
2. `beam_1prompt`: Beam search + 1 prompt
3. `beam_5prompts`: Beam search + 5 prompts
4. `nucleus_5prompts`: Nucleus sampling + 5 prompts

**주요 메서드**:
```python
generate_qa_paraphrases(combined_text, model, tokenizer)
  → List[str]  # Paraphrased QA texts
```

### 4.2 DualModelAnalyzer (analysis/paraphrase_analyzer.py)

**목적**: Full model과 Retain model을 비교하여 memorization 측정

**주요 메서드**:
```python
run_dual_model_analysis(full_model, retain_model, tokenizer, dataset, tag)
  → List[Dict]  # Results with memorization scores
```

**계산 공식**:
```python
memorization = acc_in - acc_out
simplicity = acc_in + acc_out

where:
  acc_in = Full model accuracy (trained on all data)
  acc_out = Retain model accuracy (trained without forget set)
```

### 4.3 ParaphraseVisualizer (analysis/paraphrase_visualizer.py)

**목적**: 결과 시각화

**생성 플롯**:
- Memorization vs Simplicity scatter plot
- Original vs Paraphrase comparison
- Distribution histograms

### 4.4 FullQADataset (utils.py)

**목적**: TOFU 데이터셋을 PyTorch Dataset으로 래핑

**형식**:
```python
# Input format
"[INST] {question} [/INST] {answer}"  # llama2-7b
"Question: {question}\nAnswer: {answer}"  # phi

# Output
(input_ids, labels, attention_mask)
```

================================================================================
## ⚙️ 5. 설정 파일들
================================================================================

### 5.1 model_config.yaml

모델별 형식 정의:
```yaml
llama2-7b:
  question_start_tag: "[INST] "
  question_end_tag: " [/INST]"
  answer_tag: ""

phi:
  question_start_tag: "Question: "
  question_end_tag: "\n"
  answer_tag: "Answer: "
```

### 5.2 paraphrase_analysis.yaml

Paraphrase 분석 설정:
```yaml
model_family: llama2-7b
split: forget10  # forget10, forget05, forget01

analysis:
  generation_mode: beam_1prompt
  num_paraphrases: 5
  num_beams: 4

generation:
  max_length: 256
  max_new_tokens: 256
```

### 5.3 gradient_alignment.yaml

Gradient alignment 분석 설정:
```yaml
model_family: llama2-7b
data_path: locuslab/TOFU
split: forget10

alignment:
  batch_size: 1
  num_samples: 100
```

================================================================================
## 📁 6. 데이터 흐름
================================================================================

### Input Data (TOFU Dataset)
```
locuslab/TOFU
├── forget10/  # 10% forget set
├── forget05/  # 5% forget set
├── forget01/  # 1% forget set
└── retain/    # Retain set
```

### Output Results
```
results/
├── {model}_{mode}_{split}/
│   ├── original_locuslab_TOFU_{split}_results.json
│   ├── paraphrase_locuslab_TOFU_{split}_results.json
│   ├── memorization_simplicity_original.png
│   ├── memorization_simplicity_paraphrase.png
│   └── DECODING_STRATEGY.txt
```

### JSON 결과 형식
```json
{
  "results": [
    {
      "Question": "[INST] ... [/INST]",
      "GroundTruth": "...",
      "Predicted": "...",
      "OriginalMemorization": 0.41,
      "OriginalSimplicity": 1.58,
      "ParaphraseResults": [
        {
          "paraphrased_question": "...",
          "paraphrase_memorization": 0.35,
          "memorization_difference": -0.06,
          ...
        }
      ]
    }
  ]
}
```

================================================================================
## 🔬 7. 실험 파이프라인
================================================================================

### 7.1 MIMIR MIA 실험
```bash
# 1. Configure
vim configs/mi.json

# 2. Run
python run.py --config configs/mi.json

# 3. Results
results/
└── {experiment_name}/
    ├── scores.json
    └── metrics.json
```

### 7.2 TOFU Paraphrase 실험
```bash
# 1. Configure
vim memorization/config/paraphrase_analysis.yaml

# 2. Run (from TOFU project)
cd /root/memorization_unlearn/TOFU
./run_gradient_alignment.sh

# 3. Results
results/{model}_{mode}_{split}/
├── original_*.json
├── paraphrase_*.json
└── plots/
```

================================================================================
## 🎓 8. 핵심 컨셉
================================================================================

### 8.1 Memorization Score
```
Memorization = Accuracy_in - Accuracy_out

where:
- Accuracy_in: Full model (trained on all data)
- Accuracy_out: Retain model (trained without forget set)

High memorization → Model memorized the forget set
Low memorization → Model generalized (no memorization)
```

### 8.2 Paraphrase-based Testing
```
Original Question → [Paraphrase] → Paraphrased Question
                                          ↓
                                   Does model still remember?
                                          ↓
                    If yes → True memorization (not just keyword matching)
                    If no → Shallow memorization (overfitting to wording)
```

### 8.3 Decoding Strategies
```
1. Greedy: Deterministic, fast, low diversity
2. Beam: Deterministic, medium speed, medium diversity
3. Nucleus: Stochastic, slower, high diversity

Current setup:
  - beam_1prompt with 4 beams
  - 2-word change constraint
  - min_new_tokens=10, repetition_penalty=1.1
```

================================================================================
## 🔗 9. TOFU 프로젝트와의 연동
================================================================================

### 9.1 통합 구조
```
/root/memorization_unlearn/TOFU/  (메인 프로젝트)
├── memorization/                  (심볼릭 링크 또는 복사)
│   → /root/memorization/mimir/memorization/
│
├── run_gradient_alignment.sh     (실행 스크립트)
│   → torchrun memorization/gradient_alignment_main.py
│
└── results/                      (결과 저장)
    └── {model}_{mode}_{split}/
```

### 9.2 Import 경로
```python
# TOFU 프로젝트에서
from memorization.analysis.paraphrase_analyzer import DualModelAnalyzer
from memorization.analysis.paraphrase_generator import ParaphraseGenerator
from memorization.utils import FullQADataset
```

================================================================================
## 📝 10. 주요 파일 요약
================================================================================

| 파일 | 역할 | 중요도 |
|------|------|--------|
| `paraphrase_main.py` | TOFU 연동 메인 스크립트 | ⭐⭐⭐⭐⭐ |
| `memorization/analysis/paraphrase_generator.py` | Paraphrase 생성 | ⭐⭐⭐⭐⭐ |
| `memorization/analysis/paraphrase_analyzer.py` | Dual model 분석 | ⭐⭐⭐⭐⭐ |
| `memorization/analysis/paraphrase_generation_utils.py` | 생성 유틸 | ⭐⭐⭐⭐ |
| `memorization/utils.py` | Dataset, 저장 함수 | ⭐⭐⭐⭐ |
| `memorization/config/paraphrase_analysis.yaml` | 실험 설정 | ⭐⭐⭐⭐ |
| `memorization/prompts/near_duplicate_*.txt` | Paraphrase 프롬프트 | ⭐⭐⭐ |
| `mimir/attacks/*.py` | MIA 공격 구현 | ⭐⭐⭐ |
| `run.py` | MIMIR MIA 실험 | ⭐⭐⭐ |

================================================================================
## 🚀 11. 다음 단계
================================================================================

### 현재 상태
✅ Paraphrase generator 수정 완료 (num_beams=4, min_new_tokens=10)
✅ 2-word change constraint 확인
✅ Split 정보 포함한 결과 저장 경로 설정
✅ Documentation 작성

### 실행 대기
⏳ 새 decoding strategy로 실험 실행
⏳ 결과 분석 및 비교
⏳ Memorization score 변화 확인

================================================================================
END OF CODEBASE ANALYSIS
================================================================================
