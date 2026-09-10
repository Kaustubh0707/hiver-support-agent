import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


INPUT_FILE = "data/processed/selected_brand_data.csv"
TRAINING_FILE = "data/processed/training_data.csv"

MODEL_FOLDER = "models"

VECTORIZER_FILE = os.path.join(
    MODEL_FOLDER,
    "intent_reply_vectorizer.joblib"
)

RETRIEVER_FILE = os.path.join(
    MODEL_FOLDER,
    "intent_reply_retriever.joblib"
)

DATABASE_FILE = os.path.join(
    MODEL_FOLDER,
    "intent_reply_database.csv"
)

CLASSIFIER_FILE = "models/intent_classifier.joblib"

MAX_EXAMPLES = 30000

MIN_SIMILARITY = 0.25


def classify_intent(
    message,
    classifier
):

    intent = classifier.predict(
        [message]
    )[0]

    probabilities = classifier.predict_proba(
        [message]
    )[0]

    confidence = probabilities.max()

    return intent, confidence


def build_intent_labels(
    df,
    classifier
):

    print("\nClassifying historical conversations...")

    df["intent"] = classifier.predict(
        df["customer_text"]
    )

    return df


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("INTENT-AWARE REPLY RETRIEVER")
    print("=" * 80)


    # ---------------------------------------------------------
    # 1. Check files
    # ---------------------------------------------------------

    required_files = [
        INPUT_FILE,
        CLASSIFIER_FILE
    ]

    for file in required_files:

        if not os.path.exists(file):

            print("\nERROR: Required file not found:")
            print(file)

            return


    # ---------------------------------------------------------
    # 2. Load classifier
    # ---------------------------------------------------------

    print("\nLoading intent classifier...")

    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    print("Classifier loaded.")


    # ---------------------------------------------------------
    # 3. Load historical conversations
    # ---------------------------------------------------------

    print("\nLoading AppleSupport conversations...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        "Total conversations:",
        len(df)
    )


    # ---------------------------------------------------------
    # 4. Clean data
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

    print(
        "Usable conversations:",
        len(df)
    )


    # ---------------------------------------------------------
    # 5. Sample retrieval database
    # ---------------------------------------------------------

    if len(df) > MAX_EXAMPLES:

        print(
            "\nSampling",
            MAX_EXAMPLES,
            "historical conversations..."
        )

        df = df.sample(
            n=MAX_EXAMPLES,
            random_state=42
        ).reset_index(drop=True)


    # ---------------------------------------------------------
    # 6. Assign intent to historical conversations
    # ---------------------------------------------------------

    df = build_intent_labels(
        df,
        classifier
    )


    print("\nHistorical intent distribution:")
    print("-" * 60)

    print(
        df["intent"].value_counts()
    )


    # ---------------------------------------------------------
    # 7. Create TF-IDF representation
    # ---------------------------------------------------------

    print("\nCreating TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000
    )

    X = vectorizer.fit_transform(
        df["customer_text"]
    )

    print(
        "TF-IDF matrix shape:",
        X.shape
    )


    # ---------------------------------------------------------
    # 8. Build nearest-neighbor index
    # ---------------------------------------------------------

    print("\nBuilding retrieval index...")

    retriever = NearestNeighbors(
        n_neighbors=20,
        metric="cosine",
        algorithm="brute"
    )

    retriever.fit(X)

    print(
        "Retrieval index created."
    )


    # ---------------------------------------------------------
    # 9. Save everything
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
        DATABASE_FILE,
        index=False
    )


    print("\n" + "=" * 80)
    print("INTENT-AWARE RETRIEVER CREATED")
    print("=" * 80)

    print("\nSaved files:")

    print(
        "\nVectorizer:",
        VECTORIZER_FILE
    )

    print(
        "\nRetriever:",
        RETRIEVER_FILE
    )

    print(
        "\nDatabase:",
        DATABASE_FILE
    )


    # ---------------------------------------------------------
    # 10. Test retrieval
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("TESTING INTENT-AWARE RETRIEVAL")
    print("=" * 80)


    test_messages = [

        "My iPhone battery is draining very quickly.",

        "My iPhone keeps restarting and I cannot use it.",

        "Bluetooth is not connecting to my headphones.",

        "I cannot download anything from the App Store.",

        "How do I update my iPhone?"
    ]


    for message in test_messages:

        print("\n")
        print("=" * 80)

        print(
            "CUSTOMER:",
            message
        )


        # -----------------------------------------------------
        # Predict intent
        # -----------------------------------------------------

        predicted_intent, confidence = classify_intent(
            message,
            classifier
        )

        print(
            "\nPredicted intent:",
            predicted_intent
        )

        print(
            "Classifier confidence:",
            round(confidence, 4)
        )


        # -----------------------------------------------------
        # Convert message to vector
        # -----------------------------------------------------

        query_vector = vectorizer.transform(
            [message]
        )


        # -----------------------------------------------------
        # Retrieve candidates
        # -----------------------------------------------------

        distances, indices = retriever.kneighbors(
            query_vector,
            n_neighbors=20
        )


        # -----------------------------------------------------
        # Filter candidates by intent
        # -----------------------------------------------------

        matching_cases = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            row = df.iloc[index]

            similarity = 1 - distance

            if row["intent"] != predicted_intent:
                continue

            if similarity < MIN_SIMILARITY:
                continue

            matching_cases.append(
                {
                    "similarity": similarity,
                    "customer_text": row["customer_text"],
                    "brand_text": row["brand_text"],
                    "intent": row["intent"]
                }
            )


        # -----------------------------------------------------
        # Display results
        # -----------------------------------------------------

        print(
            "\nMatching historical cases:",
            len(matching_cases)
        )


        if len(matching_cases) == 0:

            print(
                "\nNo sufficiently similar historical "
                "case found for this intent."
            )

            continue


        for rank, case in enumerate(
            matching_cases[:3],
            start=1
        ):

            print("\n")
            print(
                "CASE",
                rank
            )

            print(
                "Similarity:",
                round(
                    case["similarity"],
                    4
                )
            )

            print(
                "Intent:",
                case["intent"]
            )

            print(
                "Historical customer:",
                case["customer_text"]
            )

            print(
                "Historical AppleSupport reply:",
                case["brand_text"]
            )


if __name__ == "__main__":
    main()