<h1 align="center">Small Language Models for Privacy-Preserving Clinical Information Extraction in Low-Resource Languages</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  </a>
  <a href="https://arxiv.org/abs/2602.21374">
    <img src="https://img.shields.io/badge/arXiv-2602.21374-b31b1b.svg" alt="arXiv">
  </a>
</p>

<p align="center">
  <img src="/assets/Logo.png" alt="Logo" width="500">
</p>

---

## TL;DR

We benchmark five open-source small language models (SLMs, 1B–8B parameters) on a two-step pipeline for extracting 13 binary clinical features from 1,221 anonymized Persian palliative care transcripts — **no fine-tuning required**. Qwen2.5-7B-Instruct achieves the best overall balance (macro-F1: 0.899, MCC: 0.797). Translating Persian to English with Aya-expanse-8B improves sensitivity and completeness at a slight cost to specificity.

---

## Abstract

Extracting structured clinical information from unstructured medical transcripts in low-resource languages remains a significant challenge in healthcare natural language processing (NLP). This study evaluates a two-step pipeline combining Aya-expanse-8B as a Persian-to-English translation model with five open-source small language models (SLMs) — Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct, and Gemma-3-1B-it — for binary extraction of 13 clinical features from 1,221 anonymized Persian transcripts collected at a cancer palliative care call center. Using a few-shot prompting strategy without fine-tuning, models were assessed on macro-averaged F1-score, Matthews Correlation Coefficient (MCC), sensitivity, and specificity to account for class imbalance. Qwen2.5-7B-Instruct achieved the highest overall performance (median macro-F1: 0.899; MCC: 0.797), while Gemma-3-1B-it showed the weakest results. Larger models (7B–8B parameters) consistently outperformed smaller counterparts in sensitivity and calibration. A bilingual analysis of Aya-expanse-8B revealed that translating Persian transcripts to English improved sensitivity, reduced missing outputs, and boosted balanced metrics, though at the cost of slightly lower specificity and precision. Feature-level results showed reliable extraction of physiological symptoms across most models, whereas psychological complaints, administrative requests, and complex somatic features remained challenging. These findings establish a practical, privacy-preserving blueprint for deploying open-source SLMs in multilingual clinical NLP settings with limited infrastructure and annotation resources, and highlight the importance of jointly optimizing model scale and input language strategy for sensitive healthcare applications.


---

## Overview

![General flow image](/assets/Fig_1.png)

**Schematic overview of the study:** *The upper panel shows the dataset preprocessing, inference generation, and postprocessing, starting from 1,221 Persian palliative care phone-call transcripts, followed by translation into English, prompt construction with input–output examples, and inference using multiple small language models. The models' structured outputs are then post-processed to extract tabular data. The lower panel illustrates the multi-facet analysis framework, comparing manual extraction of 13 reference features with model-derived features through performance metrics (accuracy, sensitivity, specificity, precision, F1-score), assessment of translation effects, calibration measures (MCC, missing values), and sensitivity–specificity trade-offs.*

---

## Key Results

| Model | Median Macro-F1 | Median MCC | Median Sensitivity | Median Specificity |
|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** | **0.899** | **0.797** | 0.818 | **0.987** |
| Llama-3.1-8B-Instruct | 0.870 | 0.749 | **0.909** | 0.958 |
| Llama-3.2-3B-Instruct | 0.866 | 0.734 | 0.842 | 0.951 |
| Aya-expanse-8B (English) | 0.855 | 0.724 | 0.901 | 0.955 |
| Aya-expanse-8B (Persian) | 0.842 | 0.686 | 0.893 | 0.960 |
| Qwen2.5-1.5B-Instruct | 0.819 | 0.654 | 0.737 | 0.982 |
| Gemma-3-1B-it | 0.740 | 0.502 | 0.613 | 0.986 |

**Key findings:**
- Larger models (7B–8B params) consistently outperform smaller ones in sensitivity and overall calibration
- Translating Persian → English improves sensitivity and reduces missing outputs, with a modest drop in specificity
- Physiological symptoms (pain, fever, respiratory) are reliably extracted across most models
- Psychological complaints, administrative requests, and complex somatic features remain challenging
- All inference runs locally on a single L4 GPU (24 GB VRAM) — **no external API calls**

---



## Extracted Clinical Features

The 13 binary features extracted from each transcript:

1. Doctor's visit request
2. Psychological complaints
3. Sleep disorders
4. Loss of appetite
5. Seizures
6. Weakness and fatigue
7. Decreased level of consciousness
8. Fever
9. Respiratory complaints
10. Insurance / treatment cost issues
11. Urinary tract issues
12. Pain
13. Gastrointestinal issues

---

## Models

| Model | Parameters | HuggingFace |
|---|---|---|
| Qwen2.5-7B-Instruct | 7B | [🤗 Link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Llama-3.1-8B-Instruct | 8B | [🤗 Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Llama-3.2-3B-Instruct | 3B | [🤗 Link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Qwen2.5-1.5B-Instruct | 1.5B | [🤗 Link](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Gemma-3-1B-it | 1B | [🤗 Link](https://huggingface.co/google/gemma-3-1b-it) |
| Aya-expanse-8B *(translator)* | 8B | [🤗 Link](https://huggingface.co/CohereForAI/aya-expanse-8b) |


---

## Prompting Strategy

We use a **few-shot prompting** strategy with 3 randomly selected examples. The system prompt instructs the model to act as a clinical data extraction expert and fill a strictly structured output template with `True` / `False` values for each of the 13 features. Full prompts (English and Persian) are available in `prompts/` and in Supplementary file 2 of the paper.

**System prompt (excerpt):**
> *You are an expert in data extraction specializing in medical information. You are provided with clinical data about patients with cancer who require palliative care. Your task is to read the patient's condition and extract ONLY complications strictly into the predefined format below.*

Constraints enforced:
1. Output exactly the same structure and order of fields
2. Fill each field with `True` or `False` only
3. Do not infer or assume any information not explicitly stated
4. Do not output any text outside the provided template

---

## Data Availability

The dataset of 1,221 anonymized Persian palliative care transcripts and their manually annotated labels are available from the corresponding author upon reasonable request: **mreghafarzadeh@gmail.com**



---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{ghaffarzadehesfahani2026smalllanguagemodelsprivacypreserving,
      title={Small Language Models for Privacy-Preserving Clinical Information Extraction in Low-Resource Languages}, 
      author={Mohammadreza Ghaffarzadeh-Esfahani and Nahid Yousefian and Ebrahim Heidari-Farsani and Ali Akbar Omidvarian and Sepehr Ghahraei and Atena Farangi and AmirBahador Boroumand},
      year={2026},
      eprint={2602.21374},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.21374}, 
}
```

---

## Contact

**Mohammadreza Ghaffarzadeh-Esfahani** 

📧 mreghafarzadeh@gmail.com 


---

## License

This project is licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).
