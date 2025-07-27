

import pandas as pd
import argparse

def get_random_movie_overviews(file_path, num_movies=5):
    """
    Reads a Parquet file and prints a specified number of random movie
    titles and their plot overviews.

    Args:
        file_path (str): The path to the Parquet file.
        num_movies (int): The number of random movies to display.
    """
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        print("Please ensure the master_dataframe.parquet file exists in the 'data' directory.")
        return

    # Ensure the required columns exist
    required_columns = ['title', 'plot_overview']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The Parquet file must contain the columns: {required_columns}")
        return

    # Get a random sample of movies
    # Using n=num_movies, but safeguarding if the df is smaller than num_movies
    sample_df = df.sample(n=min(num_movies, len(df)))

    print(f"--- Here are {len(sample_df)} random movies from your dataset ---\n")

    for index, row in sample_df.iterrows():
        print("Movie to Analyze:")
        print(f"Title: {row['title']}")
        print(f"Overview: {row['plot_overview']}")
        print("Keywords: \n")
        print("-" * 20)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fetch random movie overviews from the master dataset.")
    parser.add_argument(
        "-n", "--num_movies",
        type=int,
        default=5,
        help="Number of random movies to fetch."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/Users/mariostam/Documents/Projects/thesis/llm-powered-feature-engineering/data/master_dataframe.parquet",
        help="Path to the master_dataframe.parquet file."
    )

    args = parser.parse_args()

    get_random_movie_overviews(args.file_path, args.num_movies)

