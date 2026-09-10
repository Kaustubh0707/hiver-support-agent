import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "evaluation/golden_set.csv"

OUTPUT_FOLDER = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "training_data.csv"
)

VALID_INTENTS = [
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

VALID_AUTO_HANDLE = [
    "YES",
    "NO"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("GOLDEN SET VALIDATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Golden set not found:")
        print(INPUT_FILE)

        return

    # --------------------------------------------------------
    # Load file
    # --------------------------------------------------------

    print("\nLoading golden set...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Total rows:", len(df))

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    required_columns = [
        "customer_tweet_id",
        "customer_text",
        "brand_text",
        "intent",
        "auto_handle",
        "labeling_notes"
    ]

    print("\nChecking columns...")

    missing_columns = []

    for column in required_columns:

        if column in df.columns:
            print("FOUND:", column)

        else:
            print("MISSING:", column)
            missing_columns.append(column)

    if missing_columns:

        print("\nERROR: Missing columns:")
        print(missing_columns)

        return

    # --------------------------------------------------------
    # Clean label columns
    # --------------------------------------------------------

    df["intent"] = (
        df["intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["auto_handle"] = (
        df["auto_handle"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["labeling_notes"] = (
        df["labeling_notes"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Check missing labels
    # --------------------------------------------------------

    print("\nLABEL COMPLETENESS")
    print("-" * 50)

    missing_intent = (
        df["intent"] == ""
    ).sum()

    missing_auto_handle = (
        df["auto_handle"] == ""
    ).sum()

    missing_notes = (
        df["labeling_notes"] == ""
    ).sum()

    print("Missing intent labels:", missing_intent)
    print("Missing auto-handle labels:", missing_auto_handle)
    print("Missing labeling notes:", missing_notes)

    # --------------------------------------------------------
    # Check invalid intents
    # --------------------------------------------------------

    invalid_intents = sorted(
        set(df["intent"]) - set(VALID_INTENTS) - {""}
    )

    print("\nINVALID INTENTS")
    print("-" * 50)

    if invalid_intents:

        print(invalid_intents)

    else:

        print("None")

    # --------------------------------------------------------
    # Check invalid auto-handle values
    # --------------------------------------------------------

    invalid_auto_handle = sorted(
        set(df["auto_handle"])
        - set(VALID_AUTO_HANDLE)
        - {""}
    )

    print("\nINVALID AUTO-HANDLE VALUES")
    print("-" * 50)

    if invalid_auto_handle:

        print(invalid_auto_handle)

    else:

        print("None")

    # --------------------------------------------------------
    # Find incomplete rows
    # --------------------------------------------------------

    incomplete = df[
        (df["intent"] == "")
        | (df["auto_handle"] == "")
        | (df["labeling_notes"] == "")
    ].copy()

    # --------------------------------------------------------
    # Stop if labels are incomplete
    # --------------------------------------------------------

    if len(incomplete) > 0:

        print("\n" + "=" * 80)
        print("LABELING IS NOT COMPLETE")
        print("=" * 80)

        print(
            "\nRows requiring attention:",
            len(incomplete)
        )

        print(
            "\nComplete the missing values in:"
        )

        print(INPUT_FILE)

        return

    # --------------------------------------------------------
    # Stop if invalid labels exist
    # --------------------------------------------------------

    if invalid_intents or invalid_auto_handle:

        print("\n" + "=" * 80)
        print("INVALID LABELS FOUND")
        print("=" * 80)

        print(
            "\nPlease correct the labels in:"
        )

        print(INPUT_FILE)

        return

    # --------------------------------------------------------
    # Intent distribution
    # --------------------------------------------------------

    print("\nINTENT DISTRIBUTION")
    print("-" * 50)

    print(
        df["intent"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Auto-handle distribution
    # --------------------------------------------------------

    print("\nAUTO-HANDLE DISTRIBUTION")
    print("-" * 50)

    print(
        df["auto_handle"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Save training data
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    training_df = df[
        [
            "customer_tweet_id",
            "customer_text",
            "brand_text",
            "intent",
            "auto_handle",
            "labeling_notes"
        ]
    ].copy()

    training_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("GOLDEN SET VALIDATED SUCCESSFULLY")
    print("=" * 80)

    print("\nTraining data saved to:")
    print(OUTPUT_FILE)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()