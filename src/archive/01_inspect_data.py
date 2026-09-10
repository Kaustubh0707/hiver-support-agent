import os
import pandas as pd


DATA_DIR = "data/raw"


def find_csv_files():
    """Find all CSV files inside data/raw."""
    csv_files = []

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    return csv_files


def inspect_csv(file_path):
    """Display basic information about a CSV file."""

    print("\n" + "=" * 80)
    print("FILE:", file_path)
    print("=" * 80)

    # Read only a small sample first
    df = pd.read_csv(
        file_path,
        nrows=1000,
        low_memory=False
    )

    print("\nShape of sample:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


def main():

    csv_files = find_csv_files()

    if not csv_files:
        print("No CSV files found in:", DATA_DIR)
        print("Please place the dataset inside data/raw/")
        return

    print("CSV files found:")
    for file in csv_files:
        print(" -", file)

    for file in csv_files:
        inspect_csv(file)


if __name__ == "__main__":
    main()