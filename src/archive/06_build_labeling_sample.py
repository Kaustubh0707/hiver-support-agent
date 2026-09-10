import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/processed/selected_brand_data.csv"

OUTPUT_FOLDER = "evaluation"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "golden_set.csv"
)

# Number of examples to manually label
SAMPLE_SIZE = 200

RANDOM_STATE = 42


# ============================================================
# INTENT LIST
# ============================================================

INTENTS = [
    "software_update_issue",
    "battery_issue",
    "performance_freezing",
    "app_or_itunes_issue",
    "connectivity_issue",
    "device_functionality_issue",
    "account_or_billing_issue",
    "information_or_how_to",
    "general_support",
    "other"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("BUILDING GOLDEN LABELING SAMPLE")
    print("=" * 80)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Input file not found:")
        print(INPUT_FILE)

        return

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading AppleSupport conversations...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Total conversations:", len(df))

    # --------------------------------------------------------
    # Remove empty messages
    # --------------------------------------------------------

    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["customer_text"] != ""
    ].copy()

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["customer_text"]
    )

    print(
        "Unique customer messages:",
        len(df)
    )

    # --------------------------------------------------------
    # Create balanced sample
    # --------------------------------------------------------
    #
    # We use random sampling rather than simply taking the
    # first 200 rows. This reduces selection bias.
    #
    # --------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(df)
    )

    sample_df = df.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).copy()

    # --------------------------------------------------------
    # Create labeling columns
    # --------------------------------------------------------

    sample_df["intent"] = ""

    sample_df["auto_handle"] = ""

    sample_df["labeling_notes"] = ""

    # --------------------------------------------------------
    # Keep only useful columns
    # --------------------------------------------------------

    output_df = sample_df[
        [
            "customer_tweet_id",
            "customer_text",
            "brand_text",
            "intent",
            "auto_handle",
            "labeling_notes"
        ]
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\nGolden labeling sample created.")

    print(
        "Number of examples:",
        len(output_df)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Display intent options
    # --------------------------------------------------------

    print("\nINTENT OPTIONS")
    print("=" * 50)

    for number, intent in enumerate(
        INTENTS,
        start=1
    ):

        print(
            f"{number}. {intent}"
        )

    # --------------------------------------------------------
    # Labeling instructions
    # --------------------------------------------------------

    print("\nLABELING INSTRUCTIONS")
    print("=" * 50)

    print(
        """
For every customer message in golden_set.csv:

1. Read the customer message.
2. Select exactly ONE intent.
3. Decide whether the issue can be auto-handled.
4. Write a short note explaining the decision.

Use these values for auto_handle:

YES
NO

Do not change the customer_text or brand_text columns.
"""
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\nFIRST 10 EXAMPLES TO LABEL")
    print("=" * 80)

    for index, row in output_df.head(10).iterrows():

        print(
            f"\nExample {index + 1}"
        )

        print(
            "Customer:",
            row["customer_text"]
        )

        print(
            "Historical Apple Support reply:",
            row["brand_text"]
        )

        print("-" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()