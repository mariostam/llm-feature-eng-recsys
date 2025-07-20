import os
import functions_framework
import pandas as pd
import google.generativeai as genai
from google.cloud import storage
from google.api_core import retry
import gcsfs
import pyarrow.parquet as pq
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
PROCESSED_BATCHES_PREFIX = "processed_batches/"
SAMPLE_FILENAME = "temp_sample_data.parquet"
FINAL_OUTPUT_FILENAME = "master_dataframe_with_llm_features.parquet"
SAMPLE_SIZE = 50_000
BATCH_SIZE = 1000
# Controls the number of parallel API calls.
# A higher value increases speed but also the risk of hitting rate limits.
MAX_WORKERS = 20

# --- Gemini API Configuration ---
try:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
except (ValueError, TypeError) as e:
    print(f"CRITICAL ERROR: Gemini API not configured: {e}")
    model = None

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

def generate_llm_keywords(title, overview):
    if not model:
        return "ERROR: Model not initialized"
    def is_retryable(e):
        error_str = str(e).lower()
        return "429" in error_str or "503" in error_str
    custom_retry = retry.Retry(predicate=is_retryable, initial=10.0, maximum=300.0, multiplier=2.0)
    try:
        response = model.generate_content(
            prompt_template.format(title=title, overview=overview),
            request_options={'retry': custom_retry}
        )
        return response.text.strip()
    except Exception as e:
        if "quota" in str(e).lower():
            return "ERROR: Daily quota exceeded"
        return f"ERROR: Unrecoverable API error - {str(e)}"

@functions_framework.http
def create_sample_http(request):
    request_json = request.get_json(silent=True)
    if not request_json or 'bucket' not in request_json or 'input_file' not in request_json:
        return "ERROR: 'bucket' and 'input_file' must be specified.", 400
    bucket_name = request_json['bucket']
    input_file = request_json['input_file']
    gcs = gcsfs.GCSFileSystem()
    input_path = f"gs://{bucket_name}/{input_file}"
    output_path = f"gs://{bucket_name}/{SAMPLE_FILENAME}"
    try:
        with gcs.open(input_path, 'rb') as f:
            pq_file = pq.ParquetFile(f)
            if pq_file.metadata.num_rows <= SAMPLE_SIZE:
                f.seek(0)
                sample_df = pq_file.read().to_pandas()
            else:
                num_row_groups = pq_file.num_row_groups
                rows_per_group = math.ceil(pq_file.metadata.num_rows / num_row_groups)
                groups_to_sample_float = (SAMPLE_SIZE / rows_per_group) * 1.1
                groups_to_sample = min(num_row_groups, math.ceil(groups_to_sample_float))
                random_group_indices = random.sample(range(num_row_groups), groups_to_sample)
                df_list = [pq_file.read_row_group(i).to_pandas() for i in random_group_indices]
                combined_df = pd.concat(df_list, ignore_index=True)
                sample_df = combined_df.sample(n=SAMPLE_SIZE, random_state=42).copy()
        sample_df.reset_index(drop=True, inplace=True)
        with gcs.open(output_path, 'wb') as f:
            sample_df.to_parquet(f, index=False)
        return f"Successfully created and saved sample file to {output_path}", 200
    except Exception as e:
        return f"ERROR: Failed to create sample file: {e}", 500

@functions_framework.http
def process_batch_http(request):
    request_json = request.get_json(silent=True)
    if not request_json or 'bucket' not in request_json:
        return "ERROR: 'bucket' must be specified.", 400

    bucket_name = request_json['bucket']
    storage_client = storage.Client()
    gcs = gcsfs.GCSFileSystem()
    
    try:
        sample_path = f"gs://{bucket_name}/{SAMPLE_FILENAME}"
        with gcs.open(sample_path) as f:
            sample_df = pd.read_parquet(f)
    except FileNotFoundError:
        return f"ERROR: Sample file not found. Run 'create_sample_http' first.", 404
    except Exception as e:
        return f"ERROR: Could not load sample file: {e}", 500

    processed_blobs = list(storage_client.list_blobs(bucket_name, prefix=PROCESSED_BATCHES_PREFIX))
    processed_batch_files = {blob.name for blob in processed_blobs}
    total_batches = (len(sample_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    next_batch_num = -1
    for i in range(total_batches):
        if f"{PROCESSED_BATCHES_PREFIX}batch_{i}.parquet" not in processed_batch_files:
            next_batch_num = i
            break
    
    if next_batch_num == -1:
        return "Processing already complete.", 200

    print(f"Processing batch {next_batch_num}/{total_batches-1}...")
    start_offset = next_batch_num * BATCH_SIZE
    end_offset = min(start_offset + BATCH_SIZE, len(sample_df))
    batch_df = sample_df.iloc[start_offset:end_offset].copy()

    # Process the batch in parallel to speed up keyword generation.
    results = [None] * len(batch_df)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map each API call (future) to its original row index to maintain order.
        future_to_index = {
            executor.submit(generate_llm_keywords, row['title'], row['plot_overview']): index
            for index, row in batch_df.iterrows()
        }
        
        for future in as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                result = future.result()
                # Place the result in the correct position based on its original index.
                list_pos = batch_df.index.get_loc(original_index)
                results[list_pos] = result
            except Exception as e:
                print(f"ERROR processing row {original_index}: {e}")
                list_pos = batch_df.index.get_loc(original_index)
                results[list_pos] = f"ERROR: Future failed - {e}"
    
    batch_df['llm_keywords'] = results
    print(f"Finished generating keywords for batch {next_batch_num}.")

    try:
        output_filename = f"{PROCESSED_BATCHES_PREFIX}batch_{next_batch_num}.parquet"
        output_path = f"gs://{bucket_name}/{output_filename}"
        with gcs.open(output_path, 'wb') as f:
            batch_df.to_parquet(f, index=False)
        return f"Successfully processed and saved batch {next_batch_num}.", 200
    except Exception as e:
        return f"ERROR: Failed to save batch {next_batch_num}: {e}", 500