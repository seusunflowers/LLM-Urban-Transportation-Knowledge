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

## Script Introduction
### LLMUrbanKnowledge.ipynb
This script facilitates the extraction of urban-related knowledge and hidden-state embeddings from LLMs (i.e., Llama-3.1-8B, Qwen-3-8B, and Qwen-3-14B) across multiple transit and traffic datasets. It generates textual descriptions and captures numerical representations (embeddings) for spatial, temporal, and daily urban patterns. It encapsulates the following **key functions**:
- **Model-Specific Tokenization** (tokenize_llm_inputs): Formats input text into model-specific prompt templates, such as Llama 3.1 headers or Qwen 3 chat templates with thinking mode enabled.
- **Response Decoding** (decode_llm_outputs): Post-processes raw token sequences. For Qwen 3, it specifically extracts and separates the reasoning (thinking_content) from the final content using the </think> token.
- **Embedding Extraction** (get_llm_output_embeddings): Generates the model response and extracts the hidden-state embeddings for the last generated token.
- **Automated Batch Processing** (enumerate_llm_output_embeddings): Iteratively processes spatial, temporal, and daily queries for entire datasets.

---

### run_parallel_ollama.py
This script implements a multi-GPU, parallelized inference pipeline for spatial-temporal forecasting using LLMs. It leverages the **Ollama** framework to perform zero-shot/few-shot predictions on diverse datasets, including traffic speed, subway ridership, and EV charging occupancy. The core methodology involves converting numerical time-series data and spatial relationships (neighboring nodes) into natural language prompts, allowing the LLM to reason about physical trends and temporal patterns. Its **key features** including:
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
