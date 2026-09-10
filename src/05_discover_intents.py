import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/processed/selected_brand_data.csv"

OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "intent_discovery_sample.csv"
)

# Number of conversations used for clustering
SAMPLE_SIZE = 10000

# Number of intent groups
NUMBER_OF_CLUSTERS = 10


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("APPLE SUPPORT - INTENT DISCOVERY")
    print("=" * 80)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Selected brand dataset not found.")

        print("\nExpected file:")
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
    # Check customer text
    # --------------------------------------------------------

    if "customer_text" not in df.columns:

        print("\nERROR: customer_text column not found.")

        print("\nAvailable columns:")
        print(list(df.columns))

        return

    # Remove empty messages
    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["customer_text"] != ""
    ].copy()

    print("Usable conversations:", len(df))

    # --------------------------------------------------------
    # Sample conversations
    # --------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(df)
    )

    print("\nUsing conversations for clustering:", sample_size)

    sample_df = df.sample(
        n=sample_size,
        random_state=42
    ).copy()

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print("\nCreating TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95
    )

    X = vectorizer.fit_transform(
        sample_df["customer_text"]
    )

    print("TF-IDF matrix shape:", X.shape)

    # --------------------------------------------------------
    # K-Means clustering
    # --------------------------------------------------------

    print("\nRunning K-Means clustering...")

    kmeans = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        random_state=42,
        n_init=10
    )

    sample_df["cluster"] = kmeans.fit_predict(X)

    # --------------------------------------------------------
    # Get feature names
    # --------------------------------------------------------

    feature_names = vectorizer.get_feature_names_out()

    # --------------------------------------------------------
    # Display clusters
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("DISCOVERED INTENT GROUPS")
    print("=" * 80)

    for cluster_number in range(NUMBER_OF_CLUSTERS):

        print("\n")
        print("-" * 80)
        print(
            f"CLUSTER {cluster_number} "
            f"({(sample_df['cluster'] == cluster_number).sum()} examples)"
        )
        print("-" * 80)

        # --------------------------------------------
        # Top keywords
        # --------------------------------------------

        center = kmeans.cluster_centers_[cluster_number]

        top_indices = center.argsort()[
            ::-1
        ][:15]

        keywords = [
            feature_names[index]
            for index in top_indices
        ]

        print("\nTop keywords:")
        print(", ".join(keywords))

        # --------------------------------------------
        # Example customer messages
        # --------------------------------------------

        examples = sample_df[
            sample_df["cluster"] == cluster_number
        ]["customer_text"].head(5)

        print("\nExample customer messages:")

        for example_number, text in enumerate(
            examples,
            start=1
        ):

            print(
                f"{example_number}. {text}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    output_df = sample_df[
        [
            "customer_tweet_id",
            "customer_text",
            "brand_text",
            "cluster"
        ]
    ]

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("RESULT SAVED")
    print("=" * 80)

    print("\nFile:")
    print(OUTPUT_FILE)

    print("\nIntent discovery completed successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()