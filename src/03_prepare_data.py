import os
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FOLDER = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "conversation_pairs.csv")

# ============================================================
# LOAD DATASET
# ============================================================

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("PREPARING TWITTER SUPPORT CONVERSATIONS")
    print("=" * 80)

    # Check whether the file exists
    if not os.path.exists(INPUT_FILE):
        print("\nERROR: twcs.csv was not found.")
        print("Please make sure the file is located at:")
        print(INPUT_FILE)
        return

    print("\nDataset:")
    print(INPUT_FILE)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Rows loaded:", len(df))

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    print("\nChecking required columns...")

    required_columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id"
    ]

    missing_columns = []

    for column in required_columns:

        if column in df.columns:
            print("FOUND:", column)

        else:
            print("MISSING:", column)
            missing_columns.append(column)

    if len(missing_columns) > 0:

        print("\nERROR: Required columns are missing.")
        print("Missing columns:", missing_columns)

        print("\nAvailable columns:")
        print(list(df.columns))

        return

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    print("\nCleaning tweet text...")

    df["text"] = df["text"].fillna("").astype(str)

    # Remove completely empty tweets
    df = df[df["text"].str.strip() != ""]

    # Convert inbound column to boolean
    if df["inbound"].dtype == object:

        df["inbound"] = (
            df["inbound"]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False
            })
        )

    print("Rows after cleaning:", len(df))

    # --------------------------------------------------------
    # Create tweet lookup
    # --------------------------------------------------------

    print("\nCreating tweet lookup...")

    tweet_lookup = {}

    for _, row in df.iterrows():

        tweet_id = str(row["tweet_id"])

        tweet_lookup[tweet_id] = row

    # --------------------------------------------------------
    # Create customer -> brand response pairs
    # --------------------------------------------------------

    print("\nCreating conversation pairs...")

    pairs = []

    for _, row in df.iterrows():

        # We are interested in customer tweets.
        # Inbound = True means customer message.
        if row["inbound"] is not True:
            continue

        response_id = row["response_tweet_id"]

        # Skip if there is no response
        if pd.isna(response_id):
            continue

        response_id = str(response_id)

        # Find the corresponding brand response
        if response_id not in tweet_lookup:
            continue

        response = tweet_lookup[response_id]

        # Make sure the response is an outbound/brand message
        if response["inbound"] is not False:
            continue

        pairs.append({
            "customer_tweet_id": row["tweet_id"],
            "customer_author_id": row["author_id"],
            "customer_created_at": row["created_at"],
            "customer_text": row["text"],

            "brand_tweet_id": response["tweet_id"],
            "brand_author_id": response["author_id"],
            "brand_created_at": response["created_at"],
            "brand_text": response["text"]
        })

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    pairs_df = pd.DataFrame(pairs)

    print("\nCONVERSATION PAIRS")
    print("------------------")
    print("Number of pairs:", len(pairs_df))

    if len(pairs_df) == 0:

        print("\nWARNING:")
        print("No conversation pairs were found.")

        print("\nPlease check the dataset structure.")

        return

    # --------------------------------------------------------
    # Show brand statistics
    # --------------------------------------------------------

    print("\nBRAND ACCOUNT STATISTICS")
    print("------------------------")

    brand_counts = (
        pairs_df["brand_author_id"]
        .value_counts()
        .head(20)
    )

    print(brand_counts)

    # --------------------------------------------------------
    # Save processed data
    # --------------------------------------------------------

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pairs_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved processed dataset:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\nSAMPLE CONVERSATION PAIRS")
    print("=" * 80)

    for index, row in pairs_df.head(5).iterrows():

        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nBRAND:")
        print(row["brand_text"])

        print("-" * 80)


if __name__ == "__main__":
    main()