# LLM-Powered Feature Engineering for Recommender Systems

## Project Overview

This thesis project investigates the efficacy of Large Language Models (LLMs) in generating high-quality, thematic features for recommender systems. Specifically, it explores whether LLM-generated keywords can enhance the predictive accuracy of a Factorization Machine (FM) model compared to traditional human-curated keywords. The project involved building a robust, scalable, and automated cloud-based pipeline for data processing and LLM-powered feature generation, followed by rigorous model training and evaluation.

## Core Hypothesis

The primary hypothesis for this research was a null hypothesis: that there would be no statistically significant difference in predictive accuracy (Root Mean Squared Error - RMSE) between a Factorization Machine model trained with LLM-generated thematic keywords and an identical model trained with human-created keywords.

## Key Findings

Contrary to the initial hypothesis, the experimental results demonstrated that the Factorization Machine model trained with **human-curated keywords performed statistically significantly better** (achieving a lower RMSE) than the model trained with LLM-generated keywords. This finding highlights the nuanced challenges and current limitations of directly replacing high-quality, human-curated features with LLM-generated ones in this specific recommender system context.

## Methodology Highlights

The project's methodology is structured into three interconnected modules, ensuring a rigorous and reproducible comparison between human-curated and LLM-generated features:

1.  **Data Ingestion & Unification:**
    *   Integrated MovieLens 25M (user ratings) and TMDB 5000 (movie metadata, including human-curated keywords and plot overviews) datasets.
    *   Unified data into a `master_dataframe.parquet` for efficient, columnar storage.

2.  **LLM Feature Generation Pipeline:**
    *   Developed a robust, scalable, and automated serverless pipeline on Google Cloud Platform (GCP) using Cloud Functions.
    *   Employed a precise, multi-shot prompt engineered with the R.I.S.E. framework to guide the **Gemini 2.5 Flash** model in generating thematic keywords from movie plot overviews.
    *   Implemented memory-efficient sampling (`pyarrow`) and parallel API calls (`ThreadPoolExecutor`) to handle large datasets and API rate limits.
    *   Processed movies "one prompt, one movie" to prioritize output quality and consistency for research rigor.

3.  **Model Training & Evaluation:**
    *   Utilized **Factorization Machine (FM)** models, implemented in PyTorch, known for their effectiveness with sparse data and ability to capture feature interactions.
    *   Trained two separate and independent FM models: a **Control Group** (human keywords) and an **Experimental Group** (LLM keywords).
    *   Performed systematic **hyperparameter optimization using Optuna** for both models to ensure peak performance.
    *   Ensured **reproducibility** through consistent global random seeding.
    *   Evaluated models using **Root Mean Squared Error (RMSE)** as the primary metric.
    *   Assessed statistical significance using **95% Confidence Intervals (CIs)** calculated via 10,000 bootstraps, interpreting results with a two-tailed comparison.

## Technology Stack

*   **Programming Language:** Python 3.x
*   **Data Manipulation:** pandas, NumPy, scikit-learn
*   **Machine Learning:** PyTorch (for Factorization Machine implementation)
*   **LLM Interaction:** Google Gemini (google-generativeai library)
*   **Cloud Platform:** Google Cloud Platform (Cloud Functions, Cloud Storage, Cloud Scheduler)
*   **Development Environment:** Google Colaboratory (Colab), Google AI Studio
*   **Hyperparameter Optimization:** Optuna
*   **Version Control:** Git & GitHub

## Repository Structure

*   `cloud_function/`: Contains the source code for the Google Cloud Functions (`main.py`, `requirements.txt`).
*   `data/`: Placeholder for raw and processed data files (e.g., `master_dataframe.parquet`, `final_llm_features_dataset.parquet`).
*   `notebooks/`: Jupyter notebooks for data unification (`01-data-unification.ipynb`), merging processed batches (`02-merge-batches.ipynb`), and the main model training/evaluation workflow (`training_fm.ipynb`).
*   `scripts/`: Python scripts, including `run_fm_model.py` for the core experiment.

## How to Run/Reproduce

The primary way to reproduce the model training and evaluation experiments is via Google Colaboratory:

1.  **Open `notebooks/training_fm.ipynb` in Google Colab.**
2.  **Authenticate with Google Cloud:** The notebook includes a cell for `google.colab.auth.authenticate_user()` to grant access to your GCS bucket where the data resides.
3.  **Clone the Repository:** The notebook will automatically clone this GitHub repository, ensuring you're working with the latest code.
4.  **Install Dependencies:** All necessary Python libraries will be installed within the Colab environment.
5.  **Execute Cells:** Run the cells sequentially to perform data loading, feature engineering, hyperparameter tuning, model training, and evaluation.

*Note: Running the full LLM feature generation pipeline requires deploying the Cloud Functions as described in the `thesis_project_documentation.md` and having access to the Gemini API.*

## Conclusion & Future Work

This project provides a robust framework for evaluating LLM-generated features in recommender systems. While the initial hypothesis was rejected, the findings offer valuable insights into the current state of LLM-powered feature engineering. Future work could explore more advanced prompt engineering, different LLM architectures, hybrid feature approaches, and deeper qualitative analysis of the generated keywords to understand the observed performance differences.
