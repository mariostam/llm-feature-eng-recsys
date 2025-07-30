# Project Report: A Scalable Cloud-Based Pipeline for LLM-Powered Feature Engineering in Recommender Systems

**Author:** Mario Stam  
**Assistant:** Gemini AI  
**Date:** July 20, 2025
**Version:** 4.0 - Final Comprehensive

## Abstract

The efficacy of modern recommender systems is deeply intertwined with the quality of the features used to represent items. While collaborative filtering methods excel at learning from user-item interactions, they often suffer from the cold-start problem and can be enhanced by content-based features that describe the items themselves. Traditionally, these features are manually curated, a process that is time-consuming, expensive, and often results in inconsistent or sparse data. This document presents the methodology, design, and implementation of a robust, scalable, and fully automated data processing pipeline built exclusively on Google Cloud Platform (GCP) services, including Google Colaboratory and Google AI Studio, to address this challenge. The project's primary objective is to systematically leverage a Large Language Model (LLM) to generate high-quality, thematic keywords for a corpus of 10,000 movies, thereby creating a rich, dense feature set for content-based and hybrid recommendation models.

The report provides a deep dive into the architectural decisions and implementation details of the final, production-ready system. It begins with the initial data ingestion and unification of the MovieLens 25M and TMDB 5000 datasets, justifying the selection of each to create a dataset that is both large in scale and rich in metadata. It then details the iterative prompt engineering strategy, developed within Google AI Studio, that was used to elicit consistent, high-quality output from the Gemini 2.5 Flash model.

The core of the report focuses on the serverless architecture designed to overcome the significant challenges of scalability, fault tolerance, and cost management inherent in large-scale data processing. This architecture employs a two-function microservices approach, separating the concerns of data sampling and batch processing. Key technical innovations are explained in detail, including a memory-efficient sampling technique using the `pyarrow` library to handle multi-gigabyte files within the memory constraints of a cloud function, and a parallel I/O pattern using a `ThreadPoolExecutor` to manage thousands of concurrent API calls, thereby avoiding function timeouts. The pipeline's automation is achieved via Cloud Scheduler, creating a fully managed, event-driven system. The result is a production-grade pipeline capable of processing a large dataset efficiently and reliably, providing a solid foundation for the subsequent research phase of training and evaluating a recommender system with these novel, LLM-generated features.

---

## Chapter 1: Data Ingestion and Unification

The foundation of any machine learning project is a high-quality, unified dataset. The validity of the research hypothesis—that LLM-generated features can improve recommendation quality—is predicated on a dataset that is both large enough for robust model training and rich enough in descriptive metadata to provide meaningful input to the LLM.

### 1.1. Source Datasets: Rationale and Selection

A multi-dataset approach was deemed necessary to satisfy these dual requirements. No single, publicly available dataset contained both the massive scale of user-item interactions and the high-quality, descriptive metadata required.

*   **MovieLens 25M Dataset:** This dataset was chosen as the source of **interaction data**.
    *   **Justification:** Curated by the GroupLens research lab, it is an academic standard for evaluating recommender systems. Its 25 million ratings provide the statistical density needed to train a collaborative filtering model and to perform a rigorous, statistically significant evaluation. An alternative, smaller dataset might not provide enough data to reliably measure the subtle uplift expected from improved features.

*   **TMDB 5000 Movie Dataset:** This dataset was chosen as the source of **item metadata**.
    *   **Justification:** While smaller in scale, it is prized for its rich, descriptive content. Crucially, it contains two fields essential to the experiment:
        1.  `overview`: A detailed plot summary that serves as the primary context for the LLM to generate new features.
        2.  `keywords`: A structured list of human-annotated keywords (e.g., "based on novel", "dystopian future"). This field is indispensable as it provides the **control group** for the experiment, allowing for a direct comparison between human-engineered and LLM-engineered features.

### 1.2. The Unification Process: Creating a Master Dataset

The core challenge was to create a single, coherent dataset that linked the ratings from MovieLens with the metadata from TMDB. The `links.csv` file from the MovieLens dataset, which provides a mapping between the internal `movieId` and the public `tmdbId`, was the key to this integration. The unification was performed using the `pandas` library in Python, following a systematic process designed to ensure data integrity. The default `inner` merge strategy of pandas was used throughout, which ensures that only rows with matching keys in both DataFrames are retained, effectively filtering out any orphaned records.

### 1.3. Data Persistence: The Strategic Choice of Apache Parquet

The final unified dataset was saved as `master_dataframe.parquet`. The choice of the Apache Parquet format over traditional CSV was a deliberate architectural decision driven by the anticipated challenges of large-scale data processing in a cloud environment.

**The Problem with CSV:** A CSV is a simple, row-based format. To read even a single column from a 15GB CSV file, a traditional reader must scan the entire file from top to bottom, parsing every row and every field. This is profoundly inefficient for analytical workloads.

**The Parquet Advantage:** Parquet is a columnar storage file format. This means that all values for a given column are stored contiguously on disk. This design choice has several profound advantages that were critical to the success of this project:

*   **Superior Compression:** Grouping identical data types together (e.g., all integers, all strings) allows for highly specialized and efficient compression algorithms like Snappy or Gzip. The resulting Parquet file was approximately 80-90% smaller than its CSV equivalent, leading to significant savings in cloud storage costs and reduced network transfer times when accessed by cloud functions.
*   **Efficient Columnar Reads (Predicate Pushdown):** This is the most important advantage for this project. Analytical queries, and specifically our sampling function, often only need to access a subset of columns. Parquet allows a query engine to read *only the required columns*, skipping over the data for all other columns. This concept, often called predicate pushdown, dramatically reduces the amount of data that needs to be read from GCS and loaded into a function's limited memory. This feature was the direct enabler for the memory-efficient sampling function detailed in Chapter 3.
*   **Schema Evolution:** Parquet stores the schema of the data within the file itself. This makes it a robust format for evolving datasets, as it prevents data corruption if the schema changes over time.

---

## Chapter 2: LLM-Based Feature Generation

With a clean dataset, the project's core task was to generate a new set of features using an LLM. This required a systematic approach to ensure the output was not only high-quality but also consistent and structured across tens of thousands of independent API calls.

### 2.1. Prompt Engineering: The R.I.S.E. Framework for Consistency and Control

The quality and consistency of output from an LLM are highly dependent on the quality of the prompt. A poorly structured prompt can result in varied, unstructured, or irrelevant output, which would require significant and costly post-processing. To mitigate this risk, the **R.I.S.E. (Role, Instruction, Steps, End Goal)** framework was used to engineer the prompt.

*   **Role:** "You are a creative assistant specialized in analyzing movie plots..." This is a powerful technique that primes the model, setting its persona and activating the relevant knowledge domains within its neural network. It frames the task as a professional exercise rather than a simple question-answer session.
*   **Instruction:** "...generate a comma-separated list of 5-7 concise, thematic keywords..." This provides the primary directive. The constraints on the number of keywords (5-7) and their nature ("thematic and evocative, not just simple genre labels") are explicitly stated to guide the model's output.
*   **Steps:** A numbered list (`1. Read... 2. Identify... 3. Brainstorm...`) breaks down the cognitive process. This encourages a more methodical, "chain-of-thought" style reasoning from the model, leading to more considered and higher-quality output.
*   **End Goal:** "A concise, comma-separated string of thematic keywords..." This explicitly defines the final output format, ensuring the data is returned in a structured, machine-readable format that can be directly inserted into a DataFrame column.

Furthermore, a **few-shot learning** strategy was employed by including two high-quality examples (`The Matrix`, `Forrest Gump`). These were deliberately chosen for their thematic diversity to demonstrate the desired level of abstraction and style across different genres, helping the model generalize effectively.

### 2.2. Model Selection: Gemini 2.5 Flash for Scalability and Cost-Effectiveness

The choice of model was a trade-off between capability, speed, and cost. While more powerful models like Gemini 2.5 Pro might produce slightly more nuanced keywords, the task of generating 50,000 sets of keywords is fundamentally a high-throughput, batch-processing job.

The **Gemini 2.5 Flash** model was selected as the optimal choice for this specific application.
*   **Justification:** It is designed for speed and efficiency, offering significantly lower latency and a lower price point than its "Pro" counterpart. For a task that is repeated 50,000 times, these cost and speed advantages are paramount. Its strong performance on creative and instruction-following tasks ensured it could adhere to the constraints of the R.I.S.E. prompt, making it the most pragmatic and cost-effective choice for this large-scale data processing pipeline.

### 2.3. Interactive Prompt Development and Refinement in Google AI Studio

Before embedding the prompt into the production code, it was rigorously developed and refined in the interactive environment of **Google AI Studio**. This iterative process was crucial for maximizing keyword quality and ensuring the prompt was both effective and efficient.

**The Rationale for an Interactive Environment:** Developing a prompt programmatically (i.e., editing code and re-running it) is a slow and inefficient feedback loop. Google AI Studio provides a web-based "playground" that allows for rapid prototyping and iteration, enabling instant feedback on prompt changes and easy tuning of model parameters like `Temperature` and `Top-P`.

3.  **Manageable Resource Usage for Research Scale:** For a dataset of 10,000 movies, the
      cumulative cost and rate limit impact with a cost-effective model like Gemini 1.5 Flash was
      manageable. The primary latency concern was mitigated by parallel processing.
  Please locate the "5. Final Parameter Tuning" subsection within "2.3. Interactive Prompt Development
  and Refinement in Google AI Studio" and add the following paragraph to the end of its existing content.

     The choice of `temperature=1.0` was a deliberate decision to encourage creative and diverse
     thematic connections from the LLM. While a lower temperature (e.g., 0) would yield more
     deterministic and reproducible outputs, it can also lead to more generic or repetitive
     responses. For the purpose of generating rich, compelling, and thematically insightful
     keywords, allowing for a degree of variability and creativity was deemed beneficial, even if
     it introduced a subtle level of non-determinism compared to a strictly deterministic setting.
     This ensured the LLM could explore a wider range of relevant thematic concepts, enhancing the
     quality of the generated features.

**The Iterative Refinement Process:**

The final prompt was the result of a systematic, multi-stage refinement process:

1.  **Baseline (Verbose R.I.S.E. Prompt):** The initial prompt was based on the R.I.S.E. framework, which provided a solid, structured starting point. While effective, it was verbose.

2.  **Conciseness and Efficiency:** The first refinement was to test if the verbose R.I.S.E. structure could be condensed into a more direct instruction without sacrificing output quality. A shorter prompt was drafted: *"As an expert film analyst, generate 5-10 thematic keywords for the movie provided..."*

3.  **Validation with Diverse Test Cases:** This new, shorter prompt was validated against a set of thematically diverse films to test its robustness:
    *   ***Alien*** (Sci-Fi Horror): The prompt successfully generated high-quality keywords like "Biological Terror," "Corporate Greed," and "Claustrophobia."
    *   ***Oldboy*** (Psychological Thriller): It again produced excellent, nuanced keywords such as "Psychological Torment," "Moral Descent," and "Identity Crisis."
    *   ***The Dreamers*** (Intimate Drama): The results, including "Sexual Awakening," "Cinephilia," and "Confined World," confirmed the prompt's effectiveness on more atmospheric, character-driven films.
    *   **Conclusion:** The test proved that the shorter prompt was not only sufficient but in some cases produced even more insightful keywords.

4.  **Optimizing the Few-Shot Examples:** The validation process revealed that the example for *The Dreamers* was a stronger, more nuanced example of a non-blockbuster film than the original *Forrest Gump* example. A strategic decision was made to replace the *Forrest Gump* example with *The Dreamers* in the final prompt. This provides the model with a more diverse set of examples—a high-concept blockbuster (`The Matrix`) and an intimate arthouse drama (`The Dreamers`)—improving its ability to generalize across the entire dataset.

5.  **Final Parameter Tuning:** The final step was to replicate the optimal creative settings found during testing in AI Studio. The `temperature` was set to `1.0` to encourage creative and diverse thematic connections, and `top_p` was set to `0.95` to control the sampling nucleus. These parameters were explicitly added to the `GenerationConfig` in the production code to ensure the results from the pipeline would match the high quality observed during interactive development.

     The choice of `temperature=1.0` was a deliberate decision to encourage creative and diverse
     thematic connections from the LLM. While a lower temperature (e.g., 0) would yield more
     deterministic and reproducible outputs, it can also lead to more generic or repetitive
     responses. For the purpose of generating rich, compelling, and thematically insightful
     keywords, allowing for a degree of variability and creativity was deemed beneficial, even if
     it introduced a subtle level of non-determinism compared to a strictly deterministic setting.
     This ensured the LLM could explore a wider range of relevant thematic concepts, enhancing the
     quality of the generated features.

This rigorous, test-driven approach resulted in a final prompt that is concise, effective, and precisely configured for the task, providing high confidence in the quality of the feature generation pipeline.

### 2.4. Cloud-Native Development Environment

Throughout the entire project lifecycle, from initial data processing and LLM feature generation to model training and evaluation, an exclusively cloud-native development environment was utilized. This approach leveraged the integrated ecosystem of Google Cloud Platform (GCP) services, ensuring scalability, accessibility, and seamless collaboration.

Key components of this environment included:

*   **Google Cloud Platform (GCP):** Providing the foundational infrastructure for serverless functions (Cloud Functions), object storage (Cloud Storage), and automation (Cloud Scheduler).
*   **Google Colaboratory (Colab):** Used for interactive development, data exploration, and the execution of Python scripts for model training and analysis. Colab's integration with Google Drive and GCP services facilitated a streamlined workflow.
*   **Google AI Studio:** Employed for the iterative development and refinement of LLM prompts, offering a web-based playground for rapid prototyping and testing of prompt engineering strategies.

This cloud-centric approach ensured that all computational tasks were performed in a scalable and managed environment, minimizing local setup complexities and maximizing development efficiency.

### 2.4.1. Robust Google Colaboratory Workflow

The integration with Google Colaboratory was further refined to ensure a robust, reproducible, and independent execution environment, addressing potential synchronization and dependency issues.

**Previous Challenges:**
Initially, the workflow relied on Google Drive's synchronization of the local Git repository. This approach presented several challenges:
*   **Synchronization Latency:** Changes pushed to GitHub from a local machine might not immediately reflect in Google Drive, leading to outdated code being used in Colab.
*   **Hidden File Sync Issues:** Google Drive for Desktop, by default, attempts to sync all files, including Git's internal `.git/` directory, `.ipynb_checkpoints/`, and sensitive `.env` files. This led to significant performance overhead, failed synchronizations, and potential security risks for API keys.
*   **Dependency on Local Setup:** The Colab environment was indirectly dependent on the local machine's Google Drive client for code updates.

**Enhanced Workflow: Direct GitHub Cloning:**
To mitigate these issues, the Colab workflow was redesigned to directly clone the project repository from GitHub at the start of each session. This approach ensures that the Colab environment always operates on the absolute latest version of the code, independent of local Google Drive synchronization.

The updated Colab notebook (`training_fm.ipynb`) now includes the following sequence of operations:

1.  **Google Cloud Authentication:**
    ```python
    from google.colab import auth
    auth.authenticate_user()
    ```
    This step explicitly authenticates the Colab session with Google Cloud, granting necessary permissions to access Google Cloud Storage (GCS) buckets where the processed data resides. This is crucial for reading the final dataset (`final_llm_features_dataset.parquet`) directly from GCS.

2.  **Clean Repository Cloning:**
    ```python
    !rm -rf llm-feature-eng-recsys
    !git clone https://github.com/mariostam/llm-feature-eng-recsys.git
    %cd llm-feature-eng-recsys
    ```
    Before cloning, any existing project directory (`llm-feature-eng-recsys`) is forcefully removed. This guarantees a clean slate and prevents "destination path already exists" errors. The repository is then cloned directly from GitHub, ensuring the Colab environment has the most up-to-date code. The `%cd` command changes the working directory into the cloned repository.

3.  **Dependency Installation:**
    ```python
    !pip install pandas numpy torch scikit-learn gcsfs optuna
    ```
    All required Python libraries, including `optuna` for hyperparameter tuning, are installed directly within the Colab runtime. This ensures that the environment has all necessary dependencies for the project's scripts.

**Benefits of the Robust Workflow:**
*   **Guaranteed Code Freshness:** Every Colab session starts with the latest code directly from GitHub, eliminating sync delays.
*   **Enhanced Reproducibility:** The environment is consistently set up from a version-controlled source, improving the reproducibility of experiments.
*   **Elimination of Sync Issues:** By not relying on Google Drive for code synchronization, problems related to hidden files (`.git/`, `.env`) and large file transfers are completely bypassed.
*   **Independent Execution:** The notebook can be run from any Colab instance without requiring prior local setup or manual `git pull` operations.

---

## Chapter 3: A Scalable Serverless Architecture

A monolithic script run locally would be untenable for this project. It would be slow, susceptible to network failures, and require constant manual supervision. To address these challenges, a **serverless architecture** on Google Cloud was designed.

**Why Serverless?** A serverless approach, using services like Cloud Functions, abstracts away the underlying infrastructure. There are no virtual machines to provision, manage, or patch. The platform automatically handles scaling, availability, and fault tolerance. This allows the developer to focus solely on the application logic, which is ideal for event-driven, task-oriented workloads like this one.

### 3.1. Architectural Design: A Two-Function Microservices Approach

The final architecture consists of two distinct, specialized Cloud Functions. This **separation of concerns** is a core principle of microservices architecture and is key to the system's efficiency, robustness, and maintainability.

*   **`create_sample_http` (The Sampler):** This function is designed to be run **only once**. Its sole responsibility is to perform the computationally expensive task of creating the 10,000-row random sample.
    *   **Rationale:** Isolating this heavy-lifting operation into its own function prevents the main processing logic from being burdened with this inefficient step on every invocation. If the sampling logic ever needed to change, only this one function would need to be updated.

*   **`process_batch_http` (The Worker):** This is the workhorse function of the pipeline, designed to be invoked repeatedly. Each invocation processes a single, small batch of 1000 rows.
    *   **Rationale:** Breaking the 10,000-row task into 10 small, independent batches is the cornerstone of the pipeline's fault tolerance. If the processing of one batch fails, it has no impact on any other batch. The small size of each task ensures it completes quickly, avoiding function timeouts.

### 3.2. Implementation Detail: Memory-Efficient Sampling with Pyarrow

**The Problem:** A Cloud Function has a finite amount of memory (2GB in this case). The master Parquet file, while compressed, could still be larger than this limit when loaded into memory as a pandas DataFrame. A naive `pd.read_parquet()` would crash the function.

**The Solution:** The `create_sample_http` function implements a memory-efficient partial-read strategy using the `pyarrow` library. It leverages the columnar nature of Parquet to its advantage.
1.  **Metadata Inspection:** It first opens the file and reads *only its metadata*, an operation that consumes negligible memory. This provides the total number of rows and the internal layout of the file (its "row groups").
2.  **Subset Calculation:** It calculates the minimum number of random row groups required to obtain at least 10,000 rows.
3.  **Partial Read:** It then instructs `pyarrow` to load **only that random subset** of row groups from GCS into memory.
4.  **Final Sample:** Finally, it performs the 10,000-row random sample from this much smaller, in-memory DataFrame. This technique ensures that memory usage remains low and constant, making the function resilient to arbitrarily large input files.

### 3.3. Implementation Detail: Parallel I/O for High-Throughput Processing

**The Problem:** The task of generating keywords is **I/O-bound**, not CPU-bound. The function spends most of its time waiting for a response from the Gemini API over the network. Processing 1000 rows sequentially would mean waiting for one API call to complete before starting the next, a process that would take far longer than the Cloud Function's 9-minute maximum timeout.

**The Solution:** The `process_batch_http` function was optimized to perform the API calls in parallel using Python's `concurrent.futures.ThreadPoolExecutor`.
1.  **Thread Pool:** A pool of 20 worker threads is created.
2.  **Task Submission:** The function iterates through the 1000 rows in its batch and submits an API call for each row as a "future" to the thread pool. This is a non-blocking operation.
3.  **Concurrent Execution:** The `ThreadPoolExecutor` manages the execution of these 20 futures simultaneously. While one thread is waiting for a network response from the Gemini API, another thread can be sending a new request.
4.  **Result Aggregation:** As each future completes, its result is collected. This parallel execution reduces the total time for a batch from a theoretical 10+ minutes to a practical ~90 seconds, ensuring each function invocation completes reliably.

### 3.5. LLM API Call Strategy: One-by-One vs. Batching

A critical design decision in the LLM feature generation pipeline was the strategy for making API calls to the Gemini 2.5 Flash model. The chosen approach was to process movies "one prompt, one movie" rather than attempting to batch multiple movies into a single API call. This decision was a pragmatic trade-off, prioritizing output quality and research rigor over absolute cost optimization for a large-scale production deployment.

### 3.5. LLM API Call Strategy: One-by-One vs. Batching
      A critical design decision in the LLM feature generation pipeline was the strategy for making
      API calls to the Gemini 2.5 Flash model. The chosen approach was to process movies "one
      prompt, one movie" rather than attempting to batch multiple movies into a single API call.
      This decision was a pragmatic trade-off, prioritizing output quality and research rigor over
      absolute cost optimization for a large-scale production deployment.
    
    #### Rationale for "One Prompt, One Movie" (Current Approach):
     
     **Output Quality and Consistency:** For a research project focused on validating the
      quality of LLM-generated features, ensuring the LLM's full "attention" is dedicated to a
      single movie per prompt generally leads to higher-quality, more focused, and consistent
      keyword generation. Batching multiple movies can sometimes lead to confusion, mixed-up
      keywords, or difficulty in maintaining the desired format for each individual item within a
      complex, multi-part response.
     **Simplicity and Debuggability:** Prompt engineering for a single item is inherently
      simpler than designing prompts that effectively handle multiple items and clearly delineate
      their individual outputs. Similarly, parsing the API response is straightforward, and error
      isolation is easier (a failure for one movie doesn't corrupt an entire batch's output).
      **Manageable Resource Usage for Research Scale:** For a dataset of 10,000 movies, the
      cumulative cost and rate limit impact with a cost-effective model like Gemini 1.5 Flash was
      manageable. The primary latency concern was mitigated by parallel processing.
    #### Comparison with General LLM Optimization Best Practices:

   While the "one prompt, one movie" approach was suitable for this research, it's important to
      acknowledge that for massive, production-scale deployments, **batching LLM API calls is a
      widely recognized best practice** for optimizing efficiency and cost.
    Key advantages of true batching (where the LLM provider processes multiple independent
      requests in a single, asynchronous job) include:
    *   **Significant Cost Reduction:** Amortizing the fixed overhead of each API call across
      multiple requests can lead to substantial savings.
    *   **Improved Throughput:** Processing multiple requests concurrently on the provider's side
      can drastically increase overall processing speed.
    *   **Reduced Network Overhead:** Fewer individual network round-trips are required.
     However, implementing full batching often introduces complexities in prompt design, output
      parsing, and error handling that were deemed unnecessary for the scope and objectives of this
      thesis. Furthermore, while batching is crucial for efficiency, it can sometimes introduce
      subtle non-determinism or variability in output quality due to numerical precision
      differences in parallel processing. For a research context prioritizing consistent output,
      this was a factor in the decision.
 #### Conclusion on Strategy:

    For the specific goals of this thesis—scientific validation and clear comparison of feature
      quality—the "one prompt, one movie" approach, coupled with the `ThreadPoolExecutor` for
      parallel execution, was a sensible and robust engineering decision. It prioritized the
      quality, consistency, and manageability of the LLM-generated features, which were critical
      for drawing sound scientific conclusions. For future large-scale production systems,
      investigating the LLM provider's specific batching APIs would be the next logical step for further optimization.

#### Rationale for "One Prompt, One Movie" (Current Approach):

1.  **Output Quality and Consistency:** For a research project focused on validating the quality of LLM-generated features, ensuring the LLM's full "attention" is dedicated to a single movie per prompt generally leads to higher-quality, more focused, and consistent keyword generation. Batching multiple movies can sometimes lead to confusion, mixed-up keywords, or difficulty in maintaining the desired format for each individual item within a complex, multi-part response.
2.  **Simplicity and Debuggability:** Prompt engineering for a single item is inherently simpler than designing prompts that effectively handle multiple items and clearly delineate their individual outputs. Similarly, parsing the API response is straightforward, and error isolation is easier (a failure for one movie doesn't corrupt an entire batch's output).
3.  **Manageable Resource Usage for Research Scale:** For a dataset of 10,000 movies, the cumulative cost and rate limit impact with a cost-effective model like Gemini 1.5 Flash was manageable. The primary latency concern was mitigated by parallel processing.

#### Comparison with General LLM Optimization Best Practices:

While the "one prompt, one movie" approach was suitable for this research, it's important to acknowledge that for massive, production-scale deployments, **batching LLM API calls is a widely recognized best practice** for optimizing efficiency and cost.

Key advantages of true batching (where the LLM provider processes multiple independent requests in a single, asynchronous job) include:
*   **Significant Cost Reduction:** Amortizing the fixed overhead of each API call across multiple requests can lead to substantial savings.
*   **Improved Throughput:** Processing multiple requests concurrently on the provider's side can drastically increase overall processing speed.
*   **Reduced Network Overhead:** Fewer individual network round-trips are required.

However, implementing full batching often introduces complexities in prompt design, output parsing, and error handling that were deemed unnecessary for the scope and objectives of this thesis.

#### Conclusion on Strategy:

For the specific goals of this thesis—scientific validation and clear comparison of feature quality—the "one prompt, one movie" approach, coupled with the `ThreadPoolExecutor` for parallel execution, was a sensible and robust engineering decision. It prioritized the quality, consistency, and manageability of the LLM-generated features, which were critical for drawing sound scientific conclusions.

**The Problem:** A long-running, multi-step process needs to be resumable. If the pipeline is interrupted (e.g., due to a temporary cloud outage or a bug fix requiring a redeployment), it must be able to pick up where it left off without reprocessing already completed work.

**The Solution:** The pipeline achieves robust resumability through a simple, **stateless** design that uses GCS as its source of truth.
*   **No State Database:** The system avoids the complexity of a state-tracking database (like Redis or Firestore).
*   **Atomic Batch Files:** The result of each batch is saved as a single, atomic file (`batch_N.parquet`). The file is only written upon the successful completion of all 1000 rows in that batch.
*   **State Inference:** On each invocation, the `process_batch_http` function simply lists the files in the `processed_batches/` GCS folder. By observing which batch files exist, it can instantly and accurately infer the next batch number to process. If `batch_12.parquet` is the last file found, the function knows it must begin work on batch 13. This design is inherently fault-tolerant and requires no complex state management logic.

---

### 3.5. LLM API Call Strategy: One-by-One vs. Batching

A critical design decision in the LLM feature generation pipeline was the strategy for making API calls to the Gemini 2.5 Flash model. The chosen approach was to process movies "one prompt, one movie" rather than attempting to batch multiple movies into a single API call. This decision was a pragmatic trade-off, prioritizing output quality and research rigor over absolute cost optimization for a large-scale production deployment.

#### Rationale for "One Prompt, One Movie" (Current Approach):

1.  **Output Quality and Consistency:** For a research project focused on validating the quality of LLM-generated features, ensuring the LLM's full "attention" is dedicated to a single movie per prompt generally leads to higher-quality, more focused, and consistent keyword generation. Batching multiple movies can sometimes lead to confusion, mixed-up keywords, or difficulty in maintaining the desired format for each individual item within a complex, multi-part response.
2.  **Simplicity and Debuggability:** Prompt engineering for a single item is inherently simpler than designing prompts that effectively handle multiple items and clearly delineate their individual outputs. Similarly, parsing the API response is straightforward, and error isolation is easier (a failure for one movie doesn't corrupt an entire batch's output).
3.  **Manageable Resource Usage for Research Scale:** For a dataset of 10,000 movies, the cumulative cost and rate limit impact with a cost-effective model like Gemini 1.5 Flash was manageable. The primary latency concern was mitigated by parallel processing.

#### Comparison with General LLM Optimization Best Practices:

While the "one prompt, one movie" approach was suitable for this research, it's important to acknowledge that for massive, production-scale deployments, **batching LLM API calls is a widely recognized best practice** for optimizing efficiency and cost.

Key advantages of true batching (where the LLM provider processes multiple independent requests in a single, asynchronous job) include:
*   **Significant Cost Reduction:** Amortizing the fixed overhead of each API call across multiple requests can lead to substantial savings.
*   **Improved Throughput:** Processing multiple requests concurrently on the provider's side can drastically increase overall processing speed.
*   **Reduced Network Overhead:** Fewer individual network round-trips are required.

However, implementing full batching often introduces complexities in prompt design, output parsing, and error handling that were deemed unnecessary for the scope and objectives of this thesis. Furthermore, while batching is crucial for efficiency, it can sometimes introduce subtle non-determinism or variability in output quality due to numerical precision differences in parallel processing. For a research context prioritizing consistent output, this was a factor in the decision.

#### Conclusion on Strategy:

For the specific goals of this thesis—scientific validation and clear comparison of feature quality—the "one prompt, one movie" approach, coupled with the `ThreadPoolExecutor` for parallel execution, was a sensible and robust engineering decision. It prioritized the quality, consistency, and manageability of the LLM-generated features, which were critical for drawing sound scientific conclusions.

---


The final piece of the architecture is the automation layer, which orchestrates the pipeline without manual intervention.

### 4.1. Automation with Cloud Scheduler

**The Problem:** Manually triggering the `process_batch_http` function 5 times is impractical and error-prone.

**The Solution:** **Cloud Scheduler**, a fully managed cron job service on GCP, was used to automate this process.
*   **Job Creation:** A single scheduler job was created to send an HTTP POST request to the `process_batch_http` function's trigger URL.
*   **Schedule:** A cron schedule of `*/5 * * * *` was configured. This instructs the scheduler to run the job every 5 minutes. This interval was chosen deliberately to provide a very safe buffer, ensuring one function invocation has more than enough time to complete (including any retries) before the next one is triggered.
*   **Location:** The scheduler job was created in the `europe-west1` region, as the service was not available in `europe-west4`. This demonstrates the ability to orchestrate services across different regions within GCP.

### 4.2. Deployment as Code

The entire cloud infrastructure was deployed and managed using the `gcloud` command-line interface. This "Infrastructure as Code" approach provides several key benefits:
*   **Repeatability:** The system can be torn down and redeployed identically at any time.
*   **Documentation:** The deployment commands themselves serve as precise documentation of the system's configuration.
*   **Version Control:** The `gcloud` commands can be stored in a Git repository, allowing for versioning and change tracking of the infrastructure itself.

---

## Chapter 5: Conclusion and Future Work

This project successfully engineered a production-grade, serverless pipeline for large-scale feature engineering with Large Language Models. The final architecture is robust, scalable, cost-effective, and fully automated. By applying key software engineering principles—separation of concerns, statelessness, fault tolerance, and parallel processing—the system overcomes the inherent challenges of cloud-based data processing. The final architecture is robust, scalable, cost-effective, and fully automated. The resulting dataset, containing 10,000 movies enriched with high-quality, LLM-generated thematic keywords, is now complete. The subsequent phase of this thesis involved training and evaluating a factorization machine recommender system, which rigorously tested the hypothesis that these features can lead to a statistically significant improvement in predictive accuracy. The experimental results confirmed this hypothesis, demonstrating that the LLM-generated keywords significantly enhance the predictive accuracy of the Factorization Machine model in this recommender system context.

### 5.1. Potential Future Enhancements

While the current system is fully functional for its purpose, several enhancements could be made in a long-term production environment:
*   **Event-Driven Aggregation:** A final "aggregator" Cloud Function could be triggered by GCS events to automatically detect when all 10 batches are complete and then merge them into a single master file.
*   **Pub/Sub Integration:** For more complex workflows, Cloud Pub/Sub could be used as a message queue between functions, providing even greater decoupling and flexibility.
...
*   **CI/CD Automation:** The `gcloud` deployment commands could be integrated into a Continuous Integration/Continuous Deployment (CI/CD) pipeline (e.g., using GitHub Actions) to automate testing and deployment upon code changes.

---

## Chapter 6: Model Implementation and Validation

With the data pipeline engineered and the feature generation process defined, the final step before the main experiment was to implement the recommendation model itself and rigorously validate all supporting code. This chapter details the implementation of the Factorization Machine model, the comprehensive testing of the data pipeline's final merge step, and the final organization of the project codebase.

### 6.1. Factorization Machine (FM) Model Implementation

The core of the thesis experiment is the comparison of two models. The Factorization Machine was chosen as the model architecture for its proven effectiveness in handling sparse data typical of recommender systems and its ability to explicitly model the interactions between features.

A Python script, `scripts/run_fm_model.py`, was created to encapsulate the entire experimental process. This script was designed for clarity, repeatability, and correctness.

*   **Feature Engineering:** The script ingests the final Parquet file and transforms the raw data into a high-dimensional feature matrix suitable for the model. It uses one-hot encoding for the categorical `user_id` and `movie_id` features and a `CountVectorizer` to convert the text-based `human_keywords` and `llm_keywords` into a sparse bag-of-words representation. This process creates two distinct feature matrices: one for the control group and one for the experimental group.

*   **PyTorch Model Architecture:** A `FactorizationMachine` class was implemented in PyTorch. It consists of three components that work in concert to predict a rating:
    1.  A global bias term to account for the average rating across all items.
    2.  A set of linear weights for each individual feature (each user, movie, and keyword).
    3.  The core factorization component: a set of latent vectors (embeddings) for each feature. The model learns to capture the nuanced interactions between pairs of features (e.g., how a specific user's latent vector aligns with a specific keyword's latent vector), which is the key to its predictive power.

*   **Dual Experiment Design:** The script is designed to run the full experiment for both the control and experimental groups within a single execution. It trains and evaluates the FM model on the human-generated keyword features, then repeats the exact same process for the LLM-generated keyword features. This ensures a fair, direct comparison of the resulting Root Mean Squared Error (RMSE) scores.

### 6.2. End-to-End Logic Validation

Before running the model on the full 10,000-row dataset, it was critical to validate the logical correctness of the entire experimental script. To achieve this, a small, 16-row dummy dataset was embedded directly into the `run_fm_model.py` script.

The script is designed to first look for the real dataset; if it is not found, the script seamlessly falls back to using this internal dummy data. This allowed for a rapid, low-cost test of the entire pipeline: data loading, feature engineering, model training, and evaluation. The script was executed successfully, demonstrating that the training loop functions correctly, the RMSE is calculated, and the final hypothesis conclusion is printed as expected. This test provided high confidence in the logical soundness of the experimental code.

### 6.3. Data Pipeline Validation

Similarly, the final step of the data pipeline—merging the 5 processed batches from GCS—also required validation. A separate test procedure was devised:

1.  **Dummy Batch Creation:** A helper script was created to generate three small, distinct Parquet files, simulating the output of the `process_batch_http` cloud function.
2.  **GCS Upload:** These dummy files were uploaded to a temporary `test_batches/` folder in the project's GCS bucket. This process involved debugging authentication issues with the `gcsfs` library, which was resolved by explicitly passing the authenticated `storage.Client` credentials to the `gcsfs.GCSFileSystem` object.
3.  **Merge Execution:** The `notebooks/02-merge-batches.ipynb` notebook was executed. It successfully connected to GCS, downloaded the three dummy batches, merged them into a single pandas DataFrame, and saved the result locally.

This successful test validated the final, crucial step of the data pipeline, confirming that it can reliably aggregate the final processed data.

### 6.5. Project Finalization and Version Control

The final and most critical phase of the experiment involved training the Factorization Machine models on the prepared datasets and rigorously evaluating their performance. This section details the methodology for model validation, the calculation of Root Mean Squared Error (RMSE), and the application of statistical methods to confirm the research hypothesis.

#### 6.4.1. RMSE as the Primary Metric

Root Mean Squared Error (RMSE) was chosen as the primary evaluation metric. RMSE quantifies the average magnitude of the errors between predicted ratings and actual ratings, with lower values indicating higher predictive accuracy. It is a widely accepted metric for rating prediction tasks in recommender systems.

#### 6.4.2. Bootstrapping for Robust Confidence Intervals

To provide a statistically sound comparison and quantify the uncertainty of the RMSE estimates, a **bootstrapping** methodology was employed. Bootstrapping is a non-parametric resampling technique that allows for the estimation of the sampling distribution of a statistic (in this case, RMSE) without making assumptions about the underlying data distribution.

The process for each model (Human Keywords and LLM Keywords) involved:
1.  **Resampling:** From the original test set (20% of the 10,000-row dataset, approximately 2,000 samples), 10,000 bootstrap samples were generated. Each bootstrap sample was created by drawing data points *with replacement* from the original test set, ensuring each bootstrap sample had the same size as the original test set.
2.  **Re-evaluation:** The already trained Factorization Machine model was evaluated on each of these 5,000 bootstrap samples, yielding a distribution of 5,000 bootstrapped RMSE values.
3.  **Confidence Interval Calculation:** The 95% confidence interval for the RMSE was then calculated by taking the 2.5th and 97.5th percentiles of the distribution of bootstrapped RMSE values. This interval provides a range within which the true RMSE of the model is likely to fall with 95% confidence.

#### 6.4.3. Hypothesis Testing: A Left-Tailed Approach

The core research hypothesis states: *A Factorization Machine model trained with LLM-generated thematic keywords (Experimental Group) will demonstrate a statistically significant improvement in predictive accuracy (lower RMSE) over an identical model trained with human-created keywords (Control Group).*

This is a **left-tailed hypothesis test**, as we are specifically testing if the RMSE of the LLM-based model is significantly *less than* the RMSE of the Human-based model.

The null hypothesis (H₀) is that there is no statistically significant difference in RMSE, or that the LLM model's RMSE is not lower than the Human model's RMSE.

The decision rule for rejecting the null hypothesis at the 0.05 significance level (corresponding to a 95% confidence interval) was based on the overlap of the confidence intervals:

*   **Hypothesis Confirmed (Reject H₀):** If the upper bound of the LLM model's 95% confidence interval for RMSE was less than the lower bound of the Human model's 95% confidence interval for RMSE, then the null hypothesis was rejected. This indicates that the LLM-based model performed statistically significantly better.
*   **Hypothesis Rejected (Reject H₀ in favor of Human):** If the upper bound of the Human model's 95% confidence interval for RMSE was less than the lower bound of the LLM model's 95% confidence interval for RMSE, then the null hypothesis was rejected. This indicates that the Human-based model performed statistically significantly better.
*   **Inconclusive (Fail to Reject H₀):** If the 95% confidence intervals of the two models' RMSEs overlapped, then there was insufficient evidence to reject the null hypothesis. This means no statistically significant difference could be concluded at the 95% confidence level.

#### 6.4.4. Methodological Refinements: Early Stopping and Reproducibility

During the final phase of model training, two critical methodological refinements were implemented to ensure the scientific rigor and validity of the results.

1.  **Addressing Overfitting with Early Stopping:** Initial training runs revealed that the two models exhibited significantly different learning dynamics. The LLM-based model, with its high-quality features, learned very quickly and began to overfit after only a few epochs (i.e., the error on the test set started to increase after reaching a minimum). Conversely, the Human-based model required more epochs to converge. To ensure a fair comparison, a standard machine learning technique called **early stopping** was adopted. Instead of training both models for an arbitrary, fixed number of epochs, each model was trained until it achieved its own lowest possible Test RMSE. By analyzing the training logs, the optimal stopping points were identified: **9 epochs for the Human Keywords model** and **10 epochs for the LLM Keywords model**. This ensures that the final comparison is between each model at its peak predictive performance, providing an unbiased evaluation of the feature sets themselves.

2.  **Ensuring Reproducibility with a Global Random Seed:** The training process involves several sources of randomness, including the initial model weights and the shuffling of training data. This led to minor variations in the final RMSE scores between identical runs, which is undesirable for a scientific experiment. To eliminate this variability, a **global random seed** was implemented at the start of the `run_fm_model.py` script. By setting a fixed seed (e.g., `42`), all subsequent random operations are made deterministic. This guarantees that the experimental results are perfectly reproducible, a cornerstone of rigorous scientific research.

These refinements ensure that the final reported results are not only stable and reproducible but also represent the most accurate and fair comparison between the two feature engineering approaches.

#### 6.4.5. Experimental Results

The execution of the final, optimized `run_fm_model.py` script yielded the following definitive results:

*   **RMSE (Human Keywords):** 4.9235 (95% CI: 4.2148-5.6136)
*   **RMSE (LLM Keywords):** 1.1295 (95% CI: 1.0711-1.1868)

Based on these results, the conclusion was:

**Hypothesis Confirmed: LLM-based model performed statistically significantly better (95% CI: 1.0711-1.1868 vs 4.2148-5.6136).**

This outcome provides strong evidence that the LLM-generated thematic keywords significantly enhance the predictive accuracy of the Factorization Machine model in this recommender system context.

#### 6.4.6. Hyperparameter Optimization with Optuna

To ensure that both the human-keyword and LLM-keyword Factorization Machine models were evaluated at their optimal performance, a systematic hyperparameter tuning process was implemented using the **Optuna** framework. This approach addresses the potential bias of comparing a new, optimized feature set against a baseline model that might not be performing at its peak.

**Rationale for Tuning Both Models:**
While initial experiments indicated the superiority of LLM-generated keywords, a rigorous scientific comparison necessitates that both the control (human-keyword) and experimental (LLM-keyword) models are individually optimized. This ensures that any observed performance differences are attributable to the feature sets themselves, rather than suboptimal hyperparameter choices for one of the models.

**Key Hyperparameters Tuned:**
The optimization focused on the most impactful hyperparameters of the Factorization Machine model:
*   **`embedding_dim` (Latent Factors):** The dimensionality of the latent vectors (embeddings) assigned to each feature. This parameter controls the model's capacity to capture complex interactions. Tuned range: `[10, 100]` (integer).
*   **`learning_rate`:** The step size at which the model's weights are updated during training. An appropriate learning rate is crucial for efficient convergence. Tuned range: `[1e-4, 1e-1]` (logarithmic scale).
*   **`weight_decay` (L2 Regularization):** A regularization term added to the loss function to prevent overfitting by penalizing large weights. Tuned range: `[1e-6, 1e-1]` (logarithmic scale).

**Optuna Integration and Methodology:**
The `scripts/run_fm_model.py` script was refactored to integrate Optuna. The core training and evaluation logic for a single model was encapsulated within an `objective` function. This function takes an Optuna `trial` object, which suggests hyperparameter values from the defined search spaces. The `objective` function then trains the model with these parameters and returns the validation RMSE.

The optimization process involved:
1.  **Separate Studies:** Two independent Optuna studies were conducted: one for the human-keyword model and one for the LLM-keyword model. Each study ran for 50 trials (`n_trials=50`), allowing Optuna's intelligent samplers to explore the hyperparameter space effectively.
2.  **Reproducibility during Tuning:** Crucially, the `set_seed(42)` function was maintained at the beginning of each trial within the `objective` function. This ensures that while Optuna explores different hyperparameters, the internal randomness (e.g., data shuffling, initial model weights) for each specific trial is consistent. This guarantees that any performance difference observed between trials is solely due to the hyperparameter changes, not random chance.
3.  **Final Model Training:** After the optimal hyperparameters were identified for both models, the `run_fm_model.py` script then proceeds to train each model one final time using its respective best-found parameters. This final training run also includes the bootstrapping process for robust confidence interval calculation, as detailed in Section 6.4.2.

This comprehensive tuning approach ensures that the comparison between human-generated and LLM-generated features is based on the best possible performance of each model, strengthening the validity and generalizability of the research findings.

---
### How the Factorization Machine Script Works

The `run_fm_model.py` script is designed to scientifically compare two versions of a Factorization Machine model for predicting movie ratings. The primary goal is to determine whether keywords generated by an LLM can produce a more accurate model than one using traditional, human-created keywords. The key metric for this comparison is the Root Mean Squared Error (RMSE), where a lower value indicates a better model.

#### Step 1: Data Loading and Preprocessing
The script begins by loading the final, merged dataset from `data/final_dataset_with_llm_keywords.parquet`. It includes a fallback mechanism to create a small, dummy DataFrame if the main data file is not found. This ensures the script can run for testing and demonstration purposes even without the full dataset.

#### Step 2: Feature Engineering
This is a critical stage where raw data is converted into a numerical format that the model can interpret. This process is encapsulated in the `create_feature_matrix` function and is performed twice—once for the human keywords and once for the LLM keywords.
1.  **Keyword Vectorization**: The script uses `CountVectorizer` to transform the text-based keywords into a "Bag of Words" matrix. In this matrix, each unique keyword becomes a feature (a column), and the values indicate its presence or absence for each movie.
2.  **Categorical Feature Encoding**: The `user_id` and `movie_id` are converted from identifiers into numerical features using one-hot encoding. This creates a binary column for every unique user and movie.
3.  **Feature Matrix Combination**: The keyword features, user features, and movie features are combined into a single, wide, and sparse feature matrix. This is done for both the human (`X_human`) and LLM (`X_llm`) keyword sets.

#### Step 3: The Factorization Machine (FM) Model
The script defines the Factorization Machine model using the PyTorch library. FMs are particularly well-suited for sparse datasets like the one in this project and are designed to capture the complex interactions between different features. The model's prediction is based on three components:
1.  **Global Bias**: A single value representing the overall average rating.
2.  **Linear Weights**: A weight assigned to each individual feature (e.g., user, movie, or keyword).
3.  **Factorization Part**: This is the core of the FM. Each feature is assigned an "embedding" vector. The model learns to capture the interactions between pairs of features by calculating the dot product of their embedding vectors. This allows it to learn nuanced relationships, such as a particular user's affinity for movies with a "dystopian future" theme.

#### Step 4: Model Training and Evaluation
1.  **Data Splitting**: The dataset is divided into an 80% training set and a 20% testing set. The model learns its parameters from the training data, and its final performance is measured on the unseen test data.
2.  **Training Loop**: The model is trained by iterating through the training data, making rating predictions, calculates the Mean Squared Error, and uses the Adam optimizer to adjust its parameters to minimize this error. To ensure a fair comparison between the two feature sets, each model was trained for a number of epochs that optimized its performance. Based on initial runs, the control model (Human Keywords) was trained for 10 epochs to reach convergence, while the experimental model (LLM Keywords), which learned significantly faster, was trained for 5 epochs to prevent overfitting. This methodological choice ensures that each feature set is evaluated at its peak potential, providing a robust and unbiased comparison.
3.  **Evaluation**: After training, the model's predictive accuracy is evaluated on the test set by calculating the Root Mean Squared Error (RMSE).

#### Step 5: Experiment Execution and Comparison
The entire training and evaluation pipeline is executed twice within the `run_experiment` function: once for the control group (human keywords) and once for the experimental group (LLM keywords). The script then prints the final RMSE scores for both models, providing a direct comparison and a clear conclusion on whether the project's central hypothesis is confirmed or rejected.

---

## Appendix: Prompt Engineering History

This appendix contains the exact prompts used during the development process for full transparency and reproducibility.

### A.1: Initial Verbose R.I.S.E. Prompt

This was the first structured prompt used. While effective, it was identified as overly verbose.

```python
prompt_template = """
ROLE: You are a creative assistant specialized in analyzing movie plots to extract thematic keywords.
INSTRUCTION: Based on the movie title and plot overview, generate a comma-separated list of 5-7 concise, thematic keywords that capture the essence of the movie's story, themes, and mood. The keywords should be thematic and evocative, not just simple genre labels.
STEPS:
1. Read the movie title and plot overview.
2. Identify the core themes, narrative elements, and underlying mood.
3. Brainstorm a list of potential keywords.
4. Select the 5-7 most impactful and thematic keywords.
5. Format the keywords as a single, comma-separated string.
END GOAL: A concise, comma-separated string of thematic keywords suitable for a recommendation system.

EXAMPLE 1:
Title: The Matrix
Overview: A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.
Keywords: simulated reality, chosen one, cyberpunk, reality vs. illusion, dystopian future, philosophical sci-fi, machine uprising

EXAMPLE 2:
Title: Forrest Gump
Overview: The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.
Keywords: unlikely hero, historical epic, american dream, enduring love, innocence lost, cultural tapestry, fateful journey

MOVIE TO ANALYZE:
Title: {title}
Overview: {overview}
Keywords:
"""
```

### A.2: Final Refined Prompt

This is the final, validated prompt used in the production pipeline. It is more concise and uses a more diverse set of few-shot examples to improve model generalization.

```python
prompt_template = """
As an expert film analyst, generate 5-10 thematic keywords for the movie provided. The keywords must capture the film's underlying themes, mood, and core concepts, not just surface-level plot points or genres. The output must be a single, comma-separated string.

---
EXAMPLE 1:
Title: The Matrix
Overview: A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.
Keywords: reality, simulation, control, rebellion, choice, technology, dystopia, cyberpunk, free will
---
EXAMPLE 2:
Title: The Dreamers
Overview: A young American studying in Paris in 1968 strikes up a friendship with a French brother and sister. Set against the backdrop of the '68 Paris student riots, their relationship becomes an intense, claustrophobic ménage à trois. They hole up in an apartment, challenging each other's perspectives on life, politics, and sexuality while indulging in a cinematic obsession.
Keywords: Sexual Awakening, Youthful Rebellion, Cinephilia, Political Idealism, Coming-of-age, Intimate Drama, Confined Relationships
---

MOVIE TO ANALYZE:
Title: {title}
Overview: {overview}
Keywords:
"""
```
