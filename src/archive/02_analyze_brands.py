import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = "data/raw"


# ============================================================
# FIND CSV FILE
# ============================================================

def find_csv_file():

    csv_files = []

    for root, folders, files in os.walk(DATA_FOLDER):

        for file in files:

            if file.lower().endswith(".csv"):

                csv_files.append(
                    os.path.join(root, file)
                )

    if len(csv_files) == 0:

        return None

    # For this project, use the first CSV file found.
    return csv_files[0]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):

    print("\nLoading dataset...")

    # Read a limited number of rows first.
    # The complete dataset contains millions of tweets,
    # so loading everything is unnecessary at this stage.

    df = pd.read_csv(
        file_path,
        nrows=100000,
        low_memory=False
    )

    print("Rows loaded:", len(df))

    return df


# ============================================================
# FIND BRAND COLUMN
# ============================================================

def find_brand_column(df):

    possible_columns = [
        "inbound",
        "author_id",
        "user_id",
        "company",
        "brand"
    ]

    print("\nChecking possible brand-related columns...")

    for column in possible_columns:

        if column in df.columns:

            print("Found column:", column)

    return


# ============================================================
# ANALYZE COLUMNS
# ============================================================

def analyze_columns(df):

    print("\n")
    print("=" * 80)
    print("COLUMN INFORMATION")
    print("=" * 80)

    for column in df.columns:

        print("\nColumn:", column)

        print("Data type:", df[column].dtype)

        print(
            "Unique values:",
            df[column].nunique()
        )

        # Show first few values
        print("Sample values:")

        print(
            df[column]
            .dropna()
            .head(10)
            .tolist()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("BRAND ANALYSIS")
    print("=" * 80)


    # --------------------------------------------------------
    # FIND DATASET
    # --------------------------------------------------------

    file_path = find_csv_file()

    if file_path is None:

        print("\nNo CSV file found.")

        print(
            "\nPlease put the Customer Support on Twitter "
            "dataset inside:"
        )

        print("data/raw/")

        return


    print("\nDataset found:")
    print(file_path)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_dataset(file_path)


    # --------------------------------------------------------
    # DISPLAY COLUMNS
    # --------------------------------------------------------

    print("\nColumns in dataset:")

    for column in df.columns:

        print("-", column)


    # --------------------------------------------------------
    # FIND BRAND-RELATED COLUMNS
    # --------------------------------------------------------

    find_brand_column(df)


    # --------------------------------------------------------
    # DETAILED ANALYSIS
    # --------------------------------------------------------

    analyze_columns(df)


    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BRAND ANALYSIS COMPLETED")
    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()