import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


INPUT_FILE = "data/processed/selected_brand_data.csv"

MODEL_FOLDER = "models"

VECTORIZER_FILE = os.path.join(
    MODEL_FOLDER,
    "reply_vectorizer.joblib"
)

RETRIEVER_FILE = os.path.join(
    MODEL_FOLDER,
    "reply_retriever.joblib"
)

REPLIES_FILE = os.path.join(
    MODEL_FOLDER,
    "reply_database.csv"
)

MAX_EXAMPLES = 30000


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("BUILDING HISTORICAL REPLY RETRIEVER")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Check input
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Selected brand dataset not found.")

        print("\nExpected:")
        print(INPUT_FILE)

        return


    # ---------------------------------------------------------
    # 2. Load AppleSupport conversations
    # ---------------------------------------------------------

    print("\nLoading AppleSupport conversations...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Total conversations:", len(df))


    # ---------------------------------------------------------
    # 3. Clean data
    # ---------------------------------------------------------

    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["brand_text"] = (
        df["brand_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["customer_text"] != "")
        & (df["brand_text"] != "")
    ].copy()

    df = df.drop_duplicates(
        subset=["customer_text"]
    )

    print("Usable conversations:", len(df))


    # ---------------------------------------------------------
    # 4. Limit retrieval database
    # ---------------------------------------------------------

    if len(df) > MAX_EXAMPLES:

        print(
            "\nUsing random sample of",
            MAX_EXAMPLES,
            "historical conversations."
        )

        df = df.sample(
            n=MAX_EXAMPLES,
            random_state=42
        ).reset_index(drop=True)

    else:

        print(
            "\nUsing all",
            len(df),
            "historical conversations."
        )


    # ---------------------------------------------------------
    # 5. Create TF-IDF vectors
    # ---------------------------------------------------------

    print("\nCreating TF-IDF vectors...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000
    )

    X = vectorizer.fit_transform(
        df["customer_text"]
    )

    print("TF-IDF matrix shape:", X.shape)


    # ---------------------------------------------------------
    # 6. Build nearest-neighbor retriever
    # ---------------------------------------------------------

    print("\nBuilding nearest-neighbor retriever...")

    retriever = NearestNeighbors(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute"
    )

    retriever.fit(X)

    print("Retriever created successfully.")


    # ---------------------------------------------------------
    # 7. Save model files
    # ---------------------------------------------------------

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    joblib.dump(
        retriever,
        RETRIEVER_FILE
    )

    df.to_csv(
        REPLIES_FILE,
        index=False
    )


    # ---------------------------------------------------------
    # 8. Print saved files
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("RETRIEVER CREATED SUCCESSFULLY")
    print("=" * 80)

    print("\nSaved files:")

    print("\n1. Vectorizer:")
    print(VECTORIZER_FILE)

    print("\n2. Retriever:")
    print(RETRIEVER_FILE)

    print("\n3. Historical reply database:")
    print(REPLIES_FILE)


    # ---------------------------------------------------------
    # 9. Test retrieval
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL")
    print("=" * 80)


    test_messages = [

        "My iPhone battery is draining very fast.",

        "My iPhone keeps restarting.",

        "Bluetooth will not connect.",

        "I cannot download an app.",

        "How do I update my iPhone?"
    ]


    for message in test_messages:

        print("\n")
        print("-" * 80)

        print("CUSTOMER:")
        print(message)

        query_vector = vectorizer.transform(
            [message]
        )

        distances, indices = retriever.kneighbors(
            query_vector,
            n_neighbors=3
        )

        print("\nSIMILAR HISTORICAL CONVERSATIONS:")

        for rank, (distance, index) in enumerate(
            zip(distances[0], indices[0]),
            start=1
        ):

            similarity = 1 - distance

            row = df.iloc[index]

            print("\nRank:", rank)

            print(
                "Similarity:",
                round(similarity, 4)
            )

            print(
                "Historical customer:",
                row["customer_text"]
            )

            print(
                "Historical AppleSupport reply:",
                row["brand_text"]
            )


if __name__ == "__main__":
    main()