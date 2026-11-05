# OpenLLM Pulse — Automated Exploratory Data Analysis Report

**Author:** Dawood Javed Malik (`2025-AI-401`)  
**Course:** Programming for AI — Project 1: ETL + EDA Pipeline

---

## 1. Executive Dataset Overview
- **Total Unique Models Analyzed:** 390
- **Total Features Engineered:** 24
- **Models with Extracted Parameter Counts:** 173 / 390
- **Statistical Viral Outliers (IQR Method):** 42 models
- **Mean Composite Pulse Score:** 41.03 / 100

## 2. Multi-Category Task Distribution (Option A)
| Task Category (`pipeline_tag`) | Model Count | Total Downloads | Median Likes | Mean Pulse Score |
|---|---:|---:|---:|---:|
| `sentence-similarity` | 80 | 355,880,752 | 85.0 | 38.45 |
| `text-generation` | 97 | 255,571,324 | 519.0 | 48.71 |
| `image-text-to-text` | 87 | 209,074,312 | 575.0 | 51.84 |
| `feature-extraction` | 20 | 93,868,565 | 442.5 | 44.94 |
| `translation` | 37 | 32,136,210 | 77.0 | 24.83 |
| `any-to-any` | 7 | 13,932,418 | 37.0 | 46.69 |
| `text2text-generation` | 40 | 9,695,392 | 43.5 | 21.94 |
| `text-ranking` | 2 | 3,881,340 | 312.5 | 47.72 |
| `image-to-text` | 2 | 3,010,738 | 1551.5 | 50.85 |
| `text-classification` | 3 | 2,834,703 | 169.0 | 37.32 |
| `summarization` | 7 | 1,874,104 | 234.0 | 25.07 |
| `audio-text-to-text` | 4 | 452,413 | 356.5 | 29.76 |
| `time-series-forecasting` | 3 | 109,694 | 5.0 | 9.49 |
| `automatic-speech-recognition` | 1 | 44,939 | 1463.0 | 35.57 |

## 3. Quantization & Weight Formats (Option C)
| Format | Model Count | Share (%) |
|---|---:|---:|
| Safetensors (FP16/BF16) | 226 | 57.9% |
| Standard (PyTorch/Bin) | 82 | 21.0% |
| GGUF | 48 | 12.3% |
| FP8/INT8 | 20 | 5.1% |
| AWQ | 12 | 3.1% |
| INT4 | 2 | 0.5% |

## 4. Hardware VRAM Deployment Tiers (Option F)
| Hardware Tier (FP16 VRAM) | Model Count | Share (%) |
|---|---:|---:|
| Unknown Hardware Tier | 217 | 55.6% |
| Datacenter (80GB+ VRAM) | 80 | 20.5% |
| Edge / Laptop (<=8GB VRAM) | 50 | 12.8% |
| Consumer GPU (12-24GB VRAM) | 36 | 9.2% |
| Workstation (48GB VRAM) | 7 | 1.8% |

## 5. Top 10 Models by Composite Pulse Score (Option G)
| Rank | Model ID | Task | Params (B) | FP16 VRAM (GB) | Downloads | Likes | Pulse Score |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-similarity` | N/A | N/A | 181,007,867 | 6,067 | **85.91** |
| 2 | `Qwen/Qwen3.8-27B` | `image-text-to-text` | 27.00B | 59.4 GB | 8,032,442 | 19,142 | **73.99** |
| 3 | `BAAI/bge-m3` | `sentence-similarity` | N/A | N/A | 30,325,601 | 3,963 | **73.13** |
| 4 | `openai-community/gpt2` | `text-generation` | N/A | N/A | 30,091,264 | 3,915 | **71.93** |
| 5 | `BAAI/bge-small-en-v1.5` | `feature-extraction` | N/A | N/A | 48,832,479 | 664 | **70.70** |
| 6 | `Qwen/Qwen3-0.6B` | `text-generation` | 0.60B | 1.3 GB | 22,322,522 | 1,871 | **69.98** |
| 7 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `sentence-similarity` | N/A | N/A | 27,938,959 | 1,498 | **69.67** |
| 8 | `zai-org/GLM-5.3-Flash` | `image-text-to-text` | N/A | N/A | 6,303,033 | 3,050 | **67.39** |
| 9 | `google/gemma-4-31B-it` | `image-text-to-text` | 31.00B | 68.2 GB | 7,827,380 | 4,290 | **67.31** |
| 10 | `sentence-transformers/all-mpnet-base-v2` | `sentence-similarity` | N/A | N/A | 20,327,643 | 1,404 | **66.78** |

## 6. Top 10 Viral Outliers (Downloads > Q3 + 1.5 * IQR)
| Model ID | Task | Downloads | Daily Velocity | Likes | Pulse Score |
|---|---|---:|---:|---:|---:|
| `sentence-transformers/all-MiniLM-L6-v2` | `sentence-similarity` | 181,007,867 | 106,038.6/day | 6,067 | 85.91 |
| `BAAI/bge-small-en-v1.5` | `feature-extraction` | 48,832,479 | 42,500.0/day | 664 | 70.70 |
| `BAAI/bge-m3` | `sentence-similarity` | 30,325,601 | 29,966.0/day | 3,963 | 73.13 |
| `openai-community/gpt2` | `text-generation` | 30,091,264 | 17,628.2/day | 3,915 | 71.93 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `sentence-similarity` | 27,938,959 | 16,367.3/day | 1,498 | 69.67 |
| `google-t5/t5-small` | `translation` | 22,802,650 | 13,358.3/day | 688 | 65.08 |
| `Qwen/Qwen3-0.6B` | `text-generation` | 22,322,522 | 40,148.4/day | 1,871 | 69.98 |
| `sentence-transformers/all-mpnet-base-v2` | `sentence-similarity` | 20,327,643 | 11,908.4/day | 1,404 | 66.78 |
| `Qwen/Qwen3-14B` | `text-generation` | 12,949,368 | 23,290.2/day | 533 | 62.55 |
| `nomic-ai/nomic-embed-text-v1.5` | `sentence-similarity` | 11,776,951 | 11,800.5/day | 988 | 62.65 |
