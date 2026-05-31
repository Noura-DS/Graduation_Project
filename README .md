# StanceLens 🔍
### Arabic Stance Detection — Comprehensive Model Comparison

> **Graduation Project — GP2 | Data Science Department | Umm Al-Qura University | 2025/2026**

StanceLens is an Arabic stance detection system that classifies tweets into three categories:
**Positive (إيجابي)** | **Negative (سلبي)** | **Neutral (محايد)**

The project benchmarks four model families:

| Model | Type |
|---|---|
| AraBERT (Baseline) | Frozen Embeddings + Logistic Regression |
| AraBERT (Fine-tuned) | Full Fine-tuning for Classification |
| ALLaM 7B (0/1/3-shot) | Constrained Decoding |
| LLaMA 3.1 8B (0/1/3-shot) | Constrained Decoding |
| LoRA-ALLaM | Sequence Classification (PEFT) |

---

## 📁 Repository Structure

```
StanceLens/
│
├── StanceLensGP.ipynb          # Main notebook (all models)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   └── final_with_majority_vote_UTF8.csv   # Dataset (tweets + labels)
│
└── outputs/                    # Generated after running the notebook
    ├── arabert_baseline_results.csv
    ├── arabert_finetuned_results.csv
    ├── lora_allam_results.csv
    ├── final_comparison_all_models.csv
    ├── confusion_matrices_all.png
    └── f1_comparison_chart.png
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Noura-DS
cd StanceLens
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. HuggingFace Token (Required)
Some models (ALLaM, LLaMA 3.1) require a HuggingFace token with access to gated models.

- Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Create a token with **read** access
- If running on **Google Colab**: add the token as a secret named `HF_TOKEN`
- If running locally: set it as an environment variable:
  ```bash
  export HF_TOKEN="your_token_here"
  ```

### 4. Place the Dataset
Put your dataset file in the `data/` folder:
```
data/final_with_majority_vote_UTF8.csv
```
The file must contain at least two columns: `text` and `Final_Label` (values: `Pos`, `Neg`, `Nat`).

---

## ▶️ Running the Notebook

### On Google Colab (Recommended — GPU required)
1. Upload `StanceLensGP.ipynb` to Google Colab
2. Set runtime to **GPU** (T4 or better)
3. Upload the dataset to `/content/final_with_majority_vote_UTF8.csv`
4. Add `HF_TOKEN` to Colab Secrets
5. Run all cells from top to bottom

### Locally
```bash
jupyter notebook StanceLensGP.ipynb
```
> ⚠️ A GPU with at least **16GB VRAM** is recommended for ALLaM and LLaMA models.

---

## 📊 Models Overview

### AraBERT — Baseline
Uses frozen `aubmindlab/bert-base-arabertv02` embeddings (mean pooling) fed into a Logistic Regression classifier.

### AraBERT — Fine-tuned
Full fine-tuning of AraBERT for sequence classification over 5 epochs with class-weighted loss.

### ALLaM & LLaMA 3.1 — Few-shot
Uses Constrained Decoding: the model picks the label with the highest logit among the three Arabic label tokens (`إيجابي`, `سلبي`, `محايد`). Tested in 0-shot, 1-shot, and 3-shot settings.

### LoRA-ALLaM
Parameter-efficient fine-tuning (PEFT) using LoRA adapters on ALLaM-7B for sequence classification.

---

## 📈 Results

Performance comparison of all evaluated models on the Arabic stance detection dataset:

| Model | Strategy | F1-Macro | F1-Pos | F1-Neg | F1-Nat |
|---|---|---|---|---|---|
| **AraBERT (Baseline)** | Embeddings + Logistic Regression | **0.9323** | 0.9435 | 0.9433 | 0.9100 |
| **AraBERT (Fine-tuned)** | Full Fine-tuning | 0.9228 | **0.9503** | 0.9315 | 0.8867 |
| **LoRA-ALLaM** | LoRA + Sequence Classification | 0.9125 | 0.9232 | 0.9254 | 0.8889 |
| **LLaMA-3.1 (3-shot)** | Constrained Decoding | 0.5800 | 0.7165 | 0.6966 | 0.3270 |
| **LLaMA-3.1 (1-shot)** | Constrained Decoding | 0.5465 | 0.6711 | 0.7262 | 0.2422 |
| **ALLaM (3-shot)** | Constrained Decoding | 0.3435 | 0.6154 | 0.3836 | 0.0316 |
| **ALLaM (0-shot)** | Constrained Decoding | 0.3791 | 0.6209 | 0.5165 | 0.0000 |
| **ALLaM (1-shot)** | Constrained Decoding | 0.3232 | 0.6097 | 0.3599 | 0.0000 |
| **LLaMA-3.1 (0-shot)** | Constrained Decoding | 0.2527 | 0.5857 | 0.1046 | 0.0676 |

> 🏆 **Best model:** AraBERT (Baseline) achieved the highest F1-Macro of **0.9323**, outperforming all other models including larger LLMs.

Visual outputs include confusion matrices and per-class F1 bar charts.

---

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch
- HuggingFace Transformers & PEFT
- scikit-learn
- Google Colab (recommended)

---

## 👩‍💻 Authors

Data Science Department — College of Computing — Umm Al-Qura University — 2025/2026
