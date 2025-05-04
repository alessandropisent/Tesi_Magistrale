- [LLM Evaluation for Administrative Document Analysis (Determine)](#llm-evaluation-for-administrative-document-analysis-determine)
  - [Overview](#overview)
  - [Basic Usage](#basic-usage)
  - [Dependencies](#dependencies)
    - [.env variables](#env-variables)

# LLM Evaluation for Administrative Document Analysis (Determine)

## Overview

This project contains the code and data used to test and evaluate the performance of various Large Language Models (LLMs), including OpenAI and Llama models, on the task of analyzing Italian administrative documents known as "Determine". The primary focus is on documents from the municipalities of Lucca and Olbia.

The goal is to assess how accurately LLMs can interpret these documents and answer questions based on predefined checklists.


## Basic Usage

1.  **Data Preparation:**
    * Place the raw administrative documents (potentially PDFs) in the relevant `src/txt/{City}/Raw_det/` folder.
    * Use scripts like `src/txt/pdf_2_txt.py` or city-specific conversion scripts (e.g., `src/txt/Conversione_Lucca.py`) to extract text and place it into `src/txt/{City}/determinazioni/`.
    * Ensure the ground truth checklists are correctly formatted and placed in `src/txt/{City}/checklists/` (e.g., `Lucca_Determine.csv`, `Olbia_Determine.csv`).

2.  **Running LLM Inference:**
    * Configure API keys and model parameters within the main scripts (`Lucca_LLama.py`, `Olbia_OpenAI.py`, etc.) or potentially in environment variables/config files (check script implementation).
    * Execute the desired main script (e.g., `python src/Lucca_OpenAI.py`) or use the wrapper scripts (`run_wrapper.py`, `run_wrapper_multiple.py`) to generate LLM responses. Responses will be saved in the corresponding `src/openai/.../responses/` or `src/llama/.../responses/` directories.

3.  **Evaluation:**
    * Run the evaluation scripts (e.g., `python src/Evaluator.py` or `python src/EvalChoose.py`). These scripts will compare the generated LLM responses against the ground truth checklists. Check the scripts for specific arguments they might require (e.g., paths to response files and checklist files). Evaluation outputs might be stored in the `choose` directories or other specified locations.

4.  **Analysis:**
    * Use the scripts in `src/0.Analysis/` (e.g., `python src/0.Analysis/plot_matrix.py`) to generate plots and summaries from the evaluation results.

## Dependencies

```bash
pip install -r requirements.txt
```

### .env variables

```.env
OPENAI_API_KEY=""
```

