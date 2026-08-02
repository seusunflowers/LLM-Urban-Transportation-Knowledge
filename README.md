# LLM-Urban-Transportation-Knowledge

Inspired by the impressive multi-domain knowledge presented by large language models (LLMs), this study investigates whether pretrained LLM agents possess grounded understanding of the real-world heterogeneous urban transportation systems. Multiple LLM agents and various datasets from diverse metropolitan transportation systems including road traffic, public transit, and electrified transportation are considered for comprehensive evaluation.

<img title="Spatial and temporal knowledge extraction from LLM agents" width="1790" height="822" alt="llm-spatiotemporal-knowledge-framework" src="https://github.com/user-attachments/assets/b9375a82-05df-420f-bf28-194700ee3b50" />
Fig. 1 Spatial and temporal knowledge extraction from LLM agents

## LLMs
The **Llama-3.1-8B**, **Qwen-3-8B**, and **Qwen-3-14B** agents, three competitive, open-source, lightweight, and ultra-fast pretrained LLMs, are applied in this study. Released in July 2024, experiments have revealed the superiority of the Llama-3.1-8B model over the well-known GPT-3.5-Turbo model with 20 billion parameters across a variety of tasks ([Grattafiori et al., 2024](https://doi.org/10.48550/arXiv.2407.21783)). Qwen-3-8B and Qwen-3-14B are recent state-of-the-art open-source LLMs unveiled in May 2025 ([Yang et al., 2025](https://doi.org/10.48550/arXiv.2505.09388)). In contrast to the Llama-3.1-8B model, they support “thinking mode”, allowing step-by-step reasoning before final answers for complex problems. Qwen-3-8B is generally superior to the DeepSeek-R1-Distill-Qwen-32B model, while Qwen-3-14B has exhibited comparable performance to the QwQ-32B model.

## Datasets
- **[BART](https://www.bart.gov/about/reports/ridership)**: The BART dataset includes the hourly ridership of **50 stations** in the **San Francisco Bay Area’s** rapid transit system in **2024**. Each station serves as a basic spatial unit. Since metro stations are commonly the local landmarks, their names are provided to pretrained LLM agents to obtain spatial heterogeneity insights.
- **[METR-LA](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph)**: The METR-LA dataset contains 5-minute average speed data collected from **207 urban highway sensors from March to June in 2012**, released by the **Los Angeles** Metropolitan Transportation Authority. Each sensor serves as a basic spatial unit, labeled with the name of nearby landmark. Therefore, the sensors’ names with their located highways are provided to pretrained LLM agents to obtain spatial heterogeneity insights. The temporal details are inquired at a 15-minute interval.
- **[UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV?tab=readme-ov-file)**: The UrbanEV dataset includes hourly electric vehicle charging occupancies from 17,532 piles **between September 1, 2022 and February 28, 2023**, in **Shenzhen City**, Guangdong Province, China. A total of **275 traffic analysis zones (TAZs)** serve as basic spatial units for occupancy aggregation. Since the TAZs are randomly labeled, the authors provide all the names of bus and metro stations within the corresponding TAZ to LLM agents for spatial heterogeneity insights. **The raw queries are prepared in Chinese for the dataset describing a transportation system in China.**
- **[SUBWAY-MTA](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s/about_data)**: The SUBWAY-MTA dataset provides hourly ridership data for **428 subway stations** in the **New York City** metropolitan area in **2024**. Details for obtaining LLMs’ knowledge are consistent with those for the BART dataset.
- **Demand-QD**: The Demand-QD dataset includes **trip demand data collected from May 2 to May 14, 2021, across the central urban area of Qingdao, China**. A total of **84 subdistricts** serve as the basic spatial units for trip aggregation, and their names are provided to the LLM to obtain spatial heterogeneity features. The trip data have a temporal resolution of 5 minutes, whereas the LLM is queried to infer temporal features at 15-minute intervals. **The prompts are written in Chinese** because the dataset describes a transportation system in China. Notably, **the trip data are private and were collected from Baidu Map, a mainstream location-based service provider in China. The collected data cover only a subset of the overall travel demand and are inaccessible to pretrained LLMs.**

## Script Introduction
### LLMUrbanKnowledge.ipynb
This script facilitates the extraction of urban-related knowledge and hidden-state embeddings from LLMs (using Qwen-3-14B as an example) across multiple transit and traffic datasets. It generates textual descriptions and captures numerical representations (embeddings) for spatial, temporal, and daily urban patterns. The LLM inference relies on class `HuggingFaceLlmTools`, which encapsulates following **key functions**:
- **Model-Specific Tokenization** (`tokenize_llm_inputs`): Formats input text into model-specific prompt templates, such as Llama 3.1 headers or Qwen 3 chat templates with thinking mode enabled.
- **Response Decoding** (`decode_llm_outputs`): Post-processes raw token sequences. For Qwen 3, it specifically extracts and separates the reasoning (thinking_content) from the final content using the </think> token.
- **Embedding Extraction** (`get_llm_outputs`): Generates the model response and extracts the hidden-state embeddings for the last generated token.
- **Automated Batch Processing** (`enumerate_llm_outputs`): Iteratively processes spatial, temporal, and daily queries for entire datasets.

Both temporal and daily queries focus on overall dynamic characteristics across the systems. For the datasets with unique and widely-accepted labels to describe the areas (i.e., BART, Demand-QD, UrbanEV, and SUBWAY-MTA), conveying the regional information in the queries to the LLM agents is intuitive and concise. In contrast, the METR-LA dataset is defined by a square region instead of a well-defined administrative or functional boundary. Hence, the relevant freeways and cities are enumerated in the queries. Detailed spatial, temporal, and daily query templates are as follows:

Tab. 1 Spatial query templates
| Datasets | Query Template |
| :---- | :---- |
| BART (station-based) | Please provide details about the location, train operations, ridership demand, and landmarks nearby around the `<name of the station>` Station of the San Francisco Bay Area Rapid Transit System. |
| Demand-QD (zone-based) | Please provide details about the economic level, population size, land use, industrial structure, and trip demand of `<name of the subdistrict>`, Qingdao City, Shandong Province, China. (Note that the original queries are written in Chinese) |
| METR-LA (detector-based) | Please provide details about the location, traffic demand pattern, and traffic condition on a segment of the `<name of the freeway>` Freeway around `<name of the detector>` in `<name of the city>`, `<name of the county>` County, California, USA. |
| UrbanEV (TAZ-based) | Please provide details about the details about the land use, residential composition, traffic patterns, electric vehicle friendliness, charging demand, and service levels around `<names of all bus and metro stations within the TAZ>` of Shenzhen City, Guangdong Province, China. (Note that the original queries are written in Chinese) |
| SUBWAY-MTA (station-based) | Please provide details about the location, train operations, ridership demand, and landmarks nearby about the `<name of the station>` Station in `<name of the borough>` of the New York City Subway System. |

Tab. 2 Temporal query templates
| Datasets | Query Template |
| :---- | :---- |
| BART | San Francisco Bay Area Rapid Transit System during `<start and end time of the interval>`, service and ridership. |
| Demand-QD | Trip demand at `<timestamp>` in the central urban area of Qingdao, Shandong Province, China. (Note that the original queries are written in Chinese) |
| METR-LA | Typical traffic demands and conditions at `<timestamp>` on I-110, I-210, I-405, SR-2, SR-170, I-5, SR-134, US-101 in Burbank, Glendale, La Canada-Flintridge, and Los Angeles, Los Angeles County, California, USA. |
| UrbanEV | Chang demand and service levels of the electric vehicles during `<start and end time of the interval>`, Shenzhen City, Guangdong Province, China. (Note that the original queries are written in Chinese) |
| SUBWAY-MTA | New York City Subway System during `<start and end time of the interval>`, service and ridership. |

Tab. 3 Daily query templates
| Datasets | Query Template |
| :---- | :---- |
| BART | San Francisco Bay Area Rapid Transit System on a typical `<day of week>`, service and ridership. |
| Demand-QD | Trip demand on a typical `<day of week>` in the central urban area of Qingdao, Shandong Province, China. (Note that the original queries are written in Chinese) |
| METR-LA | Typical traffic demands and conditions on a typical `<day of week>` on I-110, I-210, I-405, SR-2, SR-170, I-5, SR-134, US-101 in Burbank, Glendale, La Canada-Flintridge, and Los Angeles, Los Angeles County, California, USA. |
| UrbanEV | Chang demand and service levels of the electric vehicles on a typical `<day of week>`, Shenzhen City, Guangdong Province, China. (Note that the original queries are written in Chinese) |
| SUBWAY-MTA | New York City Subway System on a typical `<day of week>`, service and ridership. |

---

### LlmKnowledgeDecoding.ipynb
Inspired by the vast knowledge parameterized by the LLMs, we assume they can comprehensively understand the spatial and temporal profiles of urban transportation systems reported in corpora. A promising method for validation is to decode the spatial and temporal features from their hidden states. Deterministic attributes such as geographic locations, POI patterns, and time-of-day indices are considered as they contribute to the fundamental spatiotemporal profiles of transportation systems and underpin complex traffic dynamics. Linear regression models (least squares) are applied to extract the semantic information embedded in **Qwen-3-14B**’s PCA-condensed hidden states. Significant accuracy and statistical correlation between the decoded and true features would provide evidence for the assumption.

The core logic is as follows:
* **Embedding Extraction** Hidden states (typically from a Qwen3 model) are loaded from `.npz` files. We apply PCA to condense the high-dimensional LLM output (e.g., 5120 dimensions) into manageable components (e.g., 20–80 dimensions) while retaining maximum variance.
* **Feature Alignment** Real-world features are prepared as targets:
   - **Spatial**: Coordinates (X, Y) and POI distributions (via OpenStreetMap).
   - **Structural**: Graph topology represented by Node2Vec embeddings.
   - **Temporal**: Time-of-day (cosine encoding) and day-of-week.
   - **Observational**: Historical average traffic speed, ridership, or occupancy.
* **Linear Decoding** We use a linear probe to see if a simple transformation can reconstruct real-world features from LLM states. High correlation ($R_s, R_p$) indicates that the LLM has successfully "encoded" that specific urban concept in its latent space.
* **Evaluation Metrics** The performance is measured using:
   - **MAE**: Average error in the reconstructed normalized feature.
   - **Spearman ($R_s$)**: Rank correlation between decoded and real values.
   - **Pearson ($R_p$)**: Linear correlation between decoded and real values.
* **Visualizations** The script generates several plots, including:
   - **Geographical KDE Maps**: Density of decoded station locations vs. real locations.
   - **Topology Links**: Comparison between real graph adjacency and decoded correlations.
   - **Trend Comparison**: Decoded weekly/daily trends compared against real historical observations.

<img width="1009" height="444" alt="image" src="https://github.com/user-attachments/assets/a0bc9d74-b739-4aee-88f9-d6fa69ffea4e" />
Fig. 2 Linearly decoding of the geographical coordinates. a-c, Distributions of the decoded and real coordinates of the spatial units in the SUBWAY-MTA, UrbanEV, and METR-LA datasets, respectively. d, Distributions of the decoded stations in Manhattan and the Bronx boroughs in New York City, USA. e, Distributions of the decoded TAZs in Luohu and Futian districts in Shenzhen City, China. d, Distributions of the decoded detectors on US-101 and SR-134 highways in Los Angeles, USA.

<img width="1021" height="435" alt="image" src="https://github.com/user-attachments/assets/1ddaa3fe-491d-48f0-9091-cefc806c35d1" />
Fig. 3 Linearly decoding of the temporal features of the BART, SUBWAY-MTA, UrbanEV, and METR-LA datasets. a-d, Decoding cosine-encoded time-of-day indices from Qwen-3-14B’s temporal knowledge to evaluate whether it understands the continuity and periodicity in time and the heterogeneity between daytime and nighttime. e-h, Decoding network-wide observations in a typical week from Qwen-3-14B’s temporal knowledge to evaluate whether it understands the typical transportation patterns in weekdays and weekends.

---

### run_parallel_ollama.py
This script implements a multi-GPU, parallelized inference pipeline for spatial-temporal forecasting using LLMs. It leverages the **Ollama** framework to perform zero-shot/few-shot predictions on diverse datasets, including traffic speed, subway ridership, and EV charging occupancy. The core methodology involves converting numerical time-series data and spatial relationships (neighboring nodes) into natural language prompts, allowing the LLM to reason about physical trends and temporal patterns. Specifically, time index, location context, historical observations of a node, and the observations of its two nearest neighbors are encoded into the prompts and instruct Qwen-3-14B to perform multi-step prediction, with outputs formatted as a Python list to facilitate numerical calculation.

Its **key features** including:
* **Multi-Domain Support:** Pre-configured pipelines for four major datasets:
    * **METR-LA:** Highway traffic speed in Los Angeles.
    * **BART:** Hourly subway ridership in the SF Bay Area.
    * **UrbanEV:** EV charging station occupancy in Shenzhen.
    * **SubwayMTA:** Hourly subway ridership in New York City.
* **Spatial-Aware Prompting:** Automatically identifies $k$-nearest neighbors for every node to provide the LLM with spatial context.
* **Parallel Inference:** Uses Python `multiprocessing` to distribute datasets across multiple GPUs.
* **Dynamic LLM Serving:** Automatically manages local Ollama server instances with unique environment variables and ports per process.

Results are saved in a structured format containing:
* `node_id`: The specific sensor or station index.
* `hist_time_index`: The starting point of the historical window.
* `prompt`: The exact text sent to the LLM.
* `llm_content`: The raw numerical prediction returned by the model.

You can modify the model name or sampling density in the `__main__` block:
* **Model:** Change `'qwen3:14b'` to any model supported by Ollama.
* **Sample Ratio:** Adjust `sample_ratio` within each `run_` function to control the number of inference windows and manage compute costs.

> **Note:** This script uses `powershell` internally to manage environment variables for Ollama on Windows. For Linux environments, the `subprocess.Popen` command in `prompt_based_ollama_inference` should be adjusted to use `/bin/bash`.
