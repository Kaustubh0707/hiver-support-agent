import os
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/processed/conversation_pairs.csv"

OUTPUT_FOLDER = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "selected_brand_data.csv"
)

SELECTED_BRAND = "AppleSupport"


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("BRAND SELECTION")
    print("=" * 80)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Processed conversation file not found.")

        print("\nExpected file:")
        print(INPUT_FILE)

        print("\nPlease run:")
        print("python src/03_prepare_data.py")

        return

    # --------------------------------------------------------
    # Load conversations
    # --------------------------------------------------------

    print("\nLoading conversation pairs...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Total conversation pairs:", len(df))

    # --------------------------------------------------------
    # Check required column
    # --------------------------------------------------------

    if "brand_author_id" not in df.columns:

        print("\nERROR: brand_author_id column not found.")

        print("\nAvailable columns:")
        print(list(df.columns))

        return

    # --------------------------------------------------------
    # Show available brands
    # --------------------------------------------------------

    print("\nTOP BRANDS")
    print("-" * 50)

    brand_counts = (
        df["brand_author_id"]
        .value_counts()
        .head(20)
    )

    print(brand_counts)

    # --------------------------------------------------------
    # Select AppleSupport
    # --------------------------------------------------------

    print("\nSelected brand:", SELECTED_BRAND)

    brand_df = df[
        df["brand_author_id"] == SELECTED_BRAND
    ].copy()

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    print(
        "AppleSupport conversation pairs:",
        len(brand_df)
    )

    if len(brand_df) == 0:

        print("\nERROR:")
        print("Selected brand was not found.")

        return

    # --------------------------------------------------------
    # Remove duplicate conversations
    # --------------------------------------------------------

    brand_df = brand_df.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    print(
        "After removing duplicate customer tweets:",
        len(brand_df)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    brand_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved selected brand dataset:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\nSAMPLE APPLESUPPORT CONVERSATIONS")
    print("=" * 80)

    for _, row in brand_df.head(5).iterrows():

        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nAPPLE SUPPORT:")
        print(row["brand_text"])

        print("-" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()