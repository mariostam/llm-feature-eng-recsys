
import pandas as pd
import os

def create_dummy_batches():
    """
    Creates three small, dummy Parquet files to simulate the output of 
    the batch processing cloud function.
    """
    output_dir = 'data/dummy_batches'
    os.makedirs(output_dir, exist_ok=True)

    # Dummy Batch 1
    df1 = pd.DataFrame({
        'movie_id': [101, 102],
        'title': ['Movie A', 'Movie B'],
        'llm_keywords': ['theme1, theme2', 'theme3, theme4']
    })
    df1.to_parquet(os.path.join(output_dir, 'dummy_batch_0.parquet'))

    # Dummy Batch 2
    df2 = pd.DataFrame({
        'movie_id': [103],
        'title': ['Movie C'],
        'llm_keywords': ['theme5, theme6']
    })
    df2.to_parquet(os.path.join(output_dir, 'dummy_batch_1.parquet'))

    # Dummy Batch 3
    df3 = pd.DataFrame({
        'movie_id': [104, 105, 106],
        'title': ['Movie D', 'Movie E', 'Movie F'],
        'llm_keywords': ['theme7, theme8', 'theme9, theme10', 'theme11, theme12']
    })
    df3.to_parquet(os.path.join(output_dir, 'dummy_batch_2.parquet'))

    print(f"Successfully created 3 dummy batch files in '{output_dir}'")

if __name__ == '__main__':
    create_dummy_batches()
