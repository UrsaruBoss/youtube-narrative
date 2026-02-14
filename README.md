# Youtube Narrative Intelligence Pipeline

## Overview

This project implements an end-to-end data analytics pipeline for monitoring and analysing the Information Environment (IE), with a focus on narrative detection and tactical messaging patterns.

The pipeline:

1. Collects public YouTube comments and metadata.
2. Cleans and preprocesses raw textual data.
3. Translates content to English (if needed).
4. Performs sentiment analysis.
5. Detects geopolitical stance using zero-shot NLI.
6. Identifies narrative tactics (propaganda, conspiracy, dehumanization, etc.).
7. Generates structured analytical summaries and cross-tab reports.

The system is designed as a modular ETL-style workflow, where each stage can be executed independently. Outputs are stored as CSV files and analytical summaries suitable for strategic review.

---

## Hardware Requirements

This pipeline performs large-scale transformer inference (sentiment, stance, tactics) and is computationally intensive.

**Recommended configuration:**

* Modern multi-core CPU (8+ cores recommended)
* 32 GB RAM (minimum 16 GB for medium datasets)
* NVIDIA GPU with CUDA support (recommended for large datasets)

  * 8 GB+ VRAM recommended

The system can run on CPU-only mode, but processing time will increase significantly for datasets above ~50k records.

For large-scale runs (100k+ messages), a dedicated GPU-enabled workstation or server is strongly recommended.

---

## Python Dependencies

Dependencies are listed in `requirements.txt`:

```
pandas==2.2.2
numpy==1.26.4
tqdm==4.66.4
python-dotenv==1.0.1

google-api-python-client==2.137.0
google-auth==2.34.0
google-auth-oauthlib==1.2.1

transformers==4.43.3
tokenizers==0.19.1
sentencepiece==0.2.0
accelerate==0.33.0
tabulate==0.9.0
```

> Note: PyTorch is required but intentionally not pinned in `requirements.txt`.
>
> Install PyTorch separately according to your hardware:
>
> * **CPU version:** follow instructions at [https://pytorch.org](https://pytorch.org) and install the CPU-only build.
> * **GPU version:** install the CUDA-compatible build for your system.

### Optional: CPU Convenience File

If you are running in CPU-only mode, you may create a helper file named `requirements-cpu.txt` with the following content:

```
torch
-r requirements.txt
```

Then install everything with:

```
pip install -r requirements-cpu.txt
```

After installing PyTorch (CPU or GPU), you can alternatively install the remaining dependencies with:

```
pip install -r requirements.txt
```

---

This project is intended as a research and analytical baseline for Information Environment monitoring. Outputs should be reviewed by analysts and are not intended to replace expert assessment.

---

## Pipeline Stages

The workflow follows a structured, modular ETL architecture. Each stage can be executed independently and produces a versioned output file for traceability and audit.

### 00 — Data Collection (`00_gather_data.py`)

* Collects public YouTube comments and metadata.
* Stores structured raw dataset in CSV format.
* Designed for reproducible ingestion and later reprocessing.

### 10 — Data Preprocessing (`10_data_preprocessing.py`)

* Cleans text (URLs, whitespace, normalization).
* Removes invalid or empty entries.
* Produces a structured intermediate dataset ready for NLP stages.

### 20 — Translation (`20_translate_data.py`)

* Translates non-English content to English.
* Ensures a unified language baseline for transformer inference.
* Preserves original text where needed.

### 30 — Sentiment Analysis (`30_sentiment_analysis.py`)

* Applies transformer-based sentiment classification.
* Outputs `sentiment_label` and confidence score.
* Includes checkpoint/resume capability for large datasets.

### 40 — Stance Detection (`40_stance_analysis.py`)

* Zero-shot NLI-based geopolitical stance classification.
* Configurable candidate labels via JSON configuration.
* Uses score and margin gating to reduce false positives.
* Outputs `stance_label` and `stance_score`.

### 50 — Narrative Tactics Detection (`50_tactics_analysis.py`)

* Zero-shot NLI detection of narrative tactics (e.g., propaganda, conspiracy, dehumanization).
* Threshold-based filtering to increase analytical precision.
* Outputs `tactic_label` and `tactic_score`.

### 90 — Summary & Reporting (`90_summary.py`)

* Generates distribution statistics (stance, tactics).
* Produces cross-tab matrices (stance × tactic).
* Exports top-confidence examples per label for analyst audit.
* Creates `summary.md` for structured reporting.

---

## System Architecture

The system follows a modular Extract–Transform–Load pattern optimized for analytical reproducibility.

```mermaid
graph LR
    A[YouTube Data Collection] --> B[Preprocessing & Cleaning]
    B --> C[Translation to English]
    C --> D[Sentiment Analysis]
    D --> E[Stance Detection (Zero-shot NLI)]
    E --> F[Tactics Detection (Zero-shot NLI)]
    F --> G[Summary & Analytical Reports]
```

Key characteristics:

* Modular stage execution
* Checkpoint/resume support
* Configurable narrative labels
* GPU-accelerated inference
* Analyst-auditable outputs

---

## Configuration Files

The pipeline behaviour is controlled through three JSON configuration files located in the `config/` directory.

### 1. `config_targets.json`

Defines data collection scope and ingestion constraints:

* Target regions
* YouTube channel handles per region
* Quota limits and year filters
* Per-channel / per-video comment limits

This allows analysts to modify geographic or media focus without changing source code.

### 2. `config_candidates_stance.json`

Defines zero-shot stance detection labels and their natural language descriptions.

* Candidate geopolitical positions (e.g., pro_west, pro_russia, anti_west, peace_call, neutral)
* Descriptive hypotheses used for NLI inference

This file enables rapid adaptation of stance taxonomy to different operational contexts.

### 3. `config_candidates_tactics.json`

Defines narrative tactic categories and their detailed descriptions.

* Conspiracy narratives
* Propaganda framing
* Dehumanization rhetoric
* Neutral baseline

The descriptions are intentionally explicit to improve zero-shot classification precision and reduce overlap.

Together, these configuration files allow the analytical framework to remain modular, transparent, and adaptable without code-level modification.

---

## Docker & Reproducibility

The project includes a `docker-compose.yml` configuration to ensure reproducible execution across environments.

### Running with Docker

Build the container:

```
docker compose build
```

Run a specific stage:

```
docker compose run --rm pipeline python 00_gather_data.py
```

Example full pipeline execution:

```
docker compose run --rm pipeline python 10_data_preprocessing.py
docker compose run --rm pipeline python 20_translate_data.py
docker compose run --rm pipeline python 30_sentiment_analysis.py
docker compose run --rm pipeline python 40_stance_analysis.py
docker compose run --rm pipeline python 50_tactics_analysis.py
docker compose run --rm pipeline python 90_summary.py
```

The Docker configuration:

* Mounts `data/` and `reports/` as persistent volumes
* Preserves HuggingFace model cache between runs
* Supports GPU execution when NVIDIA runtime is available

This ensures consistent and portable execution for large-scale analytical workloads.

---

## Complete Analytical Dataset

After executing the full pipeline (00 → 50), a consolidated dataset is produced:

`master_dataset_with_sentiment_stance_tactics.csv`

This file contains the fully processed and enriched dataset, including:

* Cleaned and translated text
* Sentiment labels and scores
* Geopolitical stance labels and scores
* Narrative tactic labels and scores
* Any intermediate metadata retained from ingestion

The consolidated dataset is designed to serve as the primary analytical baseline for further quantitative analysis, visualization, or operational assessment.

The `90_summary.py` stage generates structured analytical reports derived from this master dataset, including distribution statistics, cross-tabulations, and high-confidence examples for analyst review.

For transparency and reproducibility, the master dataset may be preserved as a reference output, while report artifacts can be regenerated at any time from the same file.

---

## Methodology & Analytical Logic

### Zero-Shot NLI Framework

Stance and tactic detection are implemented using a zero-shot Natural Language Inference (NLI) approach.

For each input text, the model evaluates whether the text *entails* a set of predefined hypotheses (defined in configuration files). Each candidate label is converted into a natural language hypothesis (e.g., "This text expresses support for NATO.").

The model outputs entailment probabilities for each hypothesis, and the highest-scoring label is selected.

### Score & Margin Gating

To reduce false positives, the system applies:

* **Minimum score threshold** (confidence floor)
* **Minimum margin threshold** (difference between top-2 labels)

If a prediction does not meet both criteria, it is reassigned to `neutral`.

This mechanism increases analytical precision and reduces over-classification in ambiguous cases.

### Length Bucketing & Performance Optimization

To improve efficiency and reduce padding overhead:

* Texts are bucketed by approximate token length
* Inference is executed in dynamically sized GPU batches
* Checkpointing enables resume after interruption

These optimizations ensure scalability for datasets exceeding 100k entries.

### Analyst-in-the-Loop Design

The pipeline does not operate as an autonomous decision system.

It is designed to:

* Surface high-confidence examples
* Provide distributional insights
* Enable manual inspection of top-scoring cases

Human analysts remain responsible for interpretation and contextual validation.

---

## Operational Relevance

This framework is designed to support Information Environment Assessment (IEA) activities by:

* Identifying dominant geopolitical narratives
* Detecting hostile or manipulative communication patterns
* Monitoring narrative shifts over time
* Supporting quantitative briefings with auditable evidence

The modular design allows rapid reconfiguration of narrative taxonomies to reflect evolving operational priorities.

Outputs are structured to support:

* Strategic communications analysis
* Audience perception monitoring
* Disinformation pattern detection
* Cross-tab assessment of stance × tactic interaction

---

## Limitations & Responsible Use

While transformer-based zero-shot models are powerful, several limitations apply:

* Predictions depend on hypothesis phrasing and label design
* Sarcasm, irony, and cultural nuance may reduce accuracy
* Zero-shot inference is not equivalent to domain-specific fine-tuning
* High-frequency neutral outputs may reflect conservative gating

This pipeline should be used as an analytical augmentation tool rather than a deterministic classification system.

All outputs should be validated through expert review before operational use.

---

## Current Dataset Snapshot

The current full analytical dataset (`master_dataset_with_sentiment_stance_tactics.csv`) contains:

* **177,430 total rows**
* **173,845 non-empty English texts (`text_en`)**

### Stance Distribution (High-Level)

* Neutral: **92.64%**
* Pro-Russia: **5.63%**
* Anti-Russia: **1.52%**
* Anti-Ukraine: **0.20%**
* Peace Call / Other: <0.05%

### Tactics Distribution (High-Level)

* Neutral: **90.30%**
* Propaganda framing: **6.90%**
* Conspiracy narratives: **2.73%**
* Dehumanization rhetoric: **0.06%**

### Cross-Analysis

The pipeline also produces stance × tactic cross-tabulations, allowing identification of:

* Which stances correlate most with propaganda or conspiracy framing
* Whether dehumanization appears clustered within specific geopolitical alignments
* Distribution of non-neutral tactical patterns within neutral stance space

---

## Reports & Audit Files

Detailed analytical outputs are automatically generated in the `reports/` directory, including:

* Full label distributions (CSV)
* Non-neutral filtered distributions
* Stance × tactic pivot tables (counts and percentages)
* Score statistics per label (mean, median, min, max)
* Top-confidence examples per stance and tactic category

The `summary.md` file provides a structured, analyst-ready overview derived from the master dataset.

The reports currently included in this repository were generated from data collected on **30 January 2026**, using the configuration parameters defined in the `config/` directory at the time of execution.

This ensures traceability between:

* Data collection scope (`config_targets.json`)
* Stance taxonomy (`config_candidates_stance.json`)
* Tactic taxonomy (`config_candidates_tactics.json`)

The full processed dataset is not included in the repository due to size and platform policy considerations. However, all analytical reports are fully reproducible by rerunning the pipeline with the same configuration files.

To regenerate the reports:

```
python scripts/90_summary.py
```

This guarantees transparency, auditability, and repeatable analytical reporting.
