import sys

import pandas as pd


def show_parquet(path: str):
    df = pd.read_parquet(path=path)
    print(df)

def show_parquet_count(path: str):
    df = pd.read_parquet(path=path)
    print(df)

    # Count the labels in the DataFrame
    label_counts = df['label'].value_counts()
    print(label_counts)


# Usage: python utils/show_parquet.py data/train_data/phase-1/prob-1/test_x.parquet
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        show_parquet_count(sys.argv[1])
    else:
        print("missing path")
        exit(1)
