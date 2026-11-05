# OpenLLM Pulse: Automated Ecosystem Intelligence, Benchmark & Scaling Pipeline

**Author:** Dawood Javed Malik , Hafiz Muhammad Ahmad , Abdullah Malik , Abdullah Hanif  
**Registration Number:** `2025-AI-401`, `2025-AI-392` , `2025-AI-397` , `2025-AI-421` 
**Course:** Programming for AI — Project 1: ETL + EDA Pipeline  
**Institution:** University of Engineering and Technology (UET) Lahore  

---

## 1. Project Overview & Problem Statement

The open-source Artificial Intelligence ecosystem on the Hugging Face Hub is growing at an exponential rate, with thousands of models released across diverse task modalities, parameter scales, quantization formats, and software licenses. However, raw repository metadata on the Hub is unstructured: parameter sizes are buried inside arbitrary string names or tags, hardware memory (VRAM) requirements are rarely stated explicitly, and raw download counts alone fail to capture recent adoption momentum.

**OpenLLM Pulse** is an automated end-to-end **Extract, Transform, Load (ETL)** and **Exploratory Data Analysis (EDA)** pipeline built in Python. It extracts live metadata across four major AI task categories from the official Hugging Face Hub REST API, enforces strict schema validation using **Pydantic**, performs regex-based feature engineering and hardware benchmarking with **Pandas** and **NumPy**, persists the structured intelligence into both a **SQLite relational database** and a **CSV flat file**, and generates high-resolution **Matplotlib/Seaborn** visualizations alongside an automated Markdown executive report.

---

## 2. Justification of Dataset, Loading Targets & EDA Objectives

### Choice of Dataset (Multi-Category Hugging Face Hub REST API — Option A)
* **Source Endpoint:** `https://huggingface.co/api/models` (Public REST API, JSON response).
* **Multi-Task Diversity:** Rather than restricting analysis to a single task, the extraction module iterates across **4 distinct AI task categories** (`text-generation`, `text2text-generation`, `image-text-to-text`, and `sentence-similarity`), pulling the top 100 models per category (~400 total records).
* **Why It Fits ETL + EDA:** Real-world API payloads from Hugging Face contain deeply nested tag arrays, missing fields, UTC ISO-8601 timestamps, and unstandardized naming conventions—making it an ideal candidate for demonstrating Pydantic validation and complex Pandas feature engineering.

### Dual Loading Targets (`sqlite3` + `csv`)
1. **Relational Database (`data/openllm_pulse.db`, table `llm_models`):** Enables structured SQL queries, fast indexed filtering by hardware VRAM constraints or license types, and seamless downstream integration.
2. **Flat File (`data/cleaned_models.csv`):** Provides immediate portability for spreadsheet inspection, version-controlled snapshots, and lightweight static dashboarding.

### Exploratory Data Analysis (EDA) Objectives
1. **Architecture & Task Dominance:** Quantify which model families (`Llama`, `Qwen`, `Mistral`, `Bert`, `Mpnet`, `Gemma`, etc.) and task pipelines dominate community adoption.
2. **Parameter Scaling & Hardware Feasibility (Option F):** Estimate FP16 and INT4 GPU memory footprints (`vram_fp16_gb`, `vram_int4_gb`) to classify models into practical deployment tiers (Edge/Laptop $\le$ 8 GB, Consumer GPU 12–24 GB, Workstation 48 GB, Datacenter 80 GB+).
3. **Quantization & Velocity Trends (Option C):** Track weight compression formats (`GGUF`, `AWQ`, `GPTQ`, `FP8`, `Safetensors`), daily download velocity (`downloads / age_days`), and statistically flag viral outliers using the Interquartile Range ($IQR = Q_3 - Q_1$).
4. **Composite Pulse Scoring (Option G):** Rank models fairly on a `0–100` scale (`pulse_score`) combining log-normalized downloads (40%), community likes (30%), daily download velocity (20%), and documentation completeness (10%).

---

## 3. Strict Technology Stack

This project strictly adheres to the standard Python library and core data science packages taught in the course:

* **Built-in Python Modules:** `os`, `json`, `re`, `sqlite3`, `datetime`, `typing`
* **External Libraries (`requirements.txt`):**
  * `requests` — HTTP GET requests with query parameters to the Hugging Face REST API.
  * `pydantic` — Data parsing, type coercion, constraint enforcement (`ge=0`), and custom `@field_validator` sanitization.
  * `pandas` & `numpy` — Deduplication, regex parsing, vectorized mathematical feature engineering, IQR outlier detection, and relational aggregation.
  * `matplotlib` & `seaborn` — High-resolution (300 DPI) statistical visualizations.

---

## 4. Project Directory Structure

```text
D:\openllm_pulse\
│
├── data\
│   ├── raw_models.json                      # Raw multi-category JSON payload backup (~400 records)
│   ├── cleaned_models.csv                   # Transformed 24-column Pandas dataset (CSV)
│   └── openllm_pulse.db                     # SQLite relational database (table: llm_models)
│
├── outputs\
│   ├── 1_architecture_distribution.png      # Bar chart: Top 10 LLM/AI architectures
│   ├── 2_size_tier_distribution.png         # Count plot: Parameter scaling tiers
│   ├── 3_license_breakdown.png              # Bar chart: Top 8 open-source licenses
│   ├── 4_downloads_vs_likes_scatter.png     # Log-log scatter plot by task & viral outlier status
│   ├── 5_correlation_heatmap.png            # Pearson correlation matrix across numerical metrics
│   ├── 6_release_timeline.png               # Time-series line chart of monthly model releases
│   ├── 7_top_pulse_leaderboard.png          # Horizontal bar chart: Top 15 models by Pulse Score
│   └── eda_summary_report.md                # Auto-generated Markdown analytical summary report
│
├── src\
│   ├── __init__.py                          # Package initializer
│   ├── extract.py                           # Multi-category Hugging Face REST API extraction (Option A)
│   ├── schema.py                            # Pydantic BaseModel validation & field sanitization
│   ├── transform.py                         # Feature engineering: Quantization, VRAM, IQR & Pulse Score (Options C, F, G)
│   ├── load.py                              # Dual storage to SQLite3 and CSV + SQL reader
│   └── eda.py                               # Statistical reporting, 7 Seaborn plots & Markdown generator (Option E)
│
├── main.py                                  # Master orchestrator script executing the full pipeline
├── requirements.txt                         # Project Python dependencies
└── README.md                                # Complete project documentation