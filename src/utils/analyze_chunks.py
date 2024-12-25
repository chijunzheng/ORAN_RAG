# analyze_chunks.py

import pandas as pd
import json
import os
import argparse

def load_chunks(file_path: str) -> pd.DataFrame:
    """
    Loads chunks from a JSON Lines (.jsonl) file into a Pandas DataFrame.
    
    Args:
        file_path (str): Path to the chunks.json file.
    
    Returns:
        pd.DataFrame: DataFrame containing chunk data.
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
    df = pd.DataFrame(chunks)
    return df

def analyze_char_counts(df: pd.DataFrame):
    """
    Analyzes character counts in the DataFrame and prints statistics.
    
    Args:
        df (pd.DataFrame): DataFrame containing chunk data with 'char_count'.
    """
    if 'char_count' not in df.columns:
        # If 'char_count' doesn't exist, compute it
        df['char_count'] = df['text'].apply(lambda x: len(x))
    
    average = df['char_count'].mean()
    minimum = df['char_count'].min()
    maximum = df['char_count'].max()
    median = df['char_count'].median()
    
    stats = {
        'Average Character Count': average,
        'Minimum Character Count': minimum,
        'Maximum Character Count': maximum,
        'Median Character Count': median
    }
    
    print("Character Count Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Analyze character counts in chunks.json")
    parser.add_argument(
        '--chunks_file',
        type=str,
        default='chunks.json',
        help='Path to the chunks.json file. (default: chunks.json)'
    )
    args = parser.parse_args()
    
    chunks_file = args.chunks_file
    
    if not os.path.exists(chunks_file):
        print(f"Error: {chunks_file} does not exist.")
        return
    
    # Load chunks into DataFrame
    df = load_chunks(chunks_file)
    
    # Analyze character counts
    stats = analyze_char_counts(df)
    
    # Save the statistics to a CSV file
    stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    stats_df.to_csv("char_count_statistics.csv", index=False)
    print("\nStatistics saved to char_count_statistics.csv")

if __name__ == "__main__":
    main()