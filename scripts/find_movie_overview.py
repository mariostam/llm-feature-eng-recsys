

import pandas as pd
import argparse

def find_movie_overviews(file_path, titles):
    """
    Reads a Parquet file and prints the title and plot overview for
    a given list of movie titles, handling duplicates.

    Args:
        file_path (str): The path to the Parquet file.
        titles (list): A list of movie titles to search for.
    """
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return

    required_columns = ['title', 'plot_overview']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The Parquet file must contain the columns: {required_columns}")
        return

    df['search_title'] = df['title'].str.lower()
    search_titles = [t.lower() for t in titles]

    found_titles = []
    for title_to_find in search_titles:
        # Find the first match for each title
        result = df[df['search_title'] == title_to_find]
        
        if not result.empty:
            first_match = result.iloc[0]
            if first_match['title'] not in found_titles:
                print("Movie to Analyze:")
                print(f"Title: {first_match['title']}")
                print(f"Overview: {first_match['plot_overview']}")
                print("Keywords: \n")
                print("-" * 20)
                found_titles.append(first_match['title'])
        else:
            print(f"--- Movie '{titles[search_titles.index(title_to_find)]}' not found in dataset. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find specific movie overviews from the master dataset.")
    parser.add_argument(
        "titles",
        nargs='+',
        help="The title(s) of the movie(s) to search for."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/Users/mariostam/Documents/Projects/thesis/llm-powered-feature-engineering/data/master_dataframe.parquet",
        help="Path to the master_dataframe.parquet file."
    )

    args = parser.parse_args()
    find_movie_overviews(args.file_path, args.titles)


