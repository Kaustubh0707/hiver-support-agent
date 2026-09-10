import os
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "data/processed/selected_brand_data.csv"
TRAINING_FILE = "data/processed/training_data.csv"

CLASSIFIER_FILE = "models/intent_classifier.joblib"
VECTORIZER_FILE = "models/intent_reply_vectorizer.joblib"
RETRIEVER_FILE = "models/intent_reply_retriever.joblib"
REPLY_DATABASE_FILE = "models/intent_reply_database.csv"

OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "support_agent_evaluation.csv"
)

SAMPLE_SIZE = 500


def decide_action(
    text,
    predicted_intent,
    confidence,
    best_similarity,
    retrieved_count
):

    text = str(text).lower()

    escalation_signals = [
        "charged",
        "refund",
        "payment",
        "account",
        "apple id",
        "password",
        "hacked",
        "stolen",
        "lost",
        "security",
        "fraud",
        "cannot access",
        "can't access",
        "not working",
        "still not working",
        "tried everything",
        "nothing works"
    ]

    if any(
        signal in text
        for signal in escalation_signals
    ):
        return "ESCALATE", "Sensitive or unresolved issue"

    if predicted_intent in [
        "general_support",
        "other"
    ]:
        return "ESCALATE", "Intent is too broad or unclear"

    if predicted_intent in [
        "performance_freezing",
        "device_functionality_issue"
    ]:
        return "ESCALATE", "Device-specific troubleshooting required"

    if confidence < 0.60:
        return "ESCALATE", "Low intent confidence"

    if retrieved_count == 0:
        return "ESCALATE", "No historical examples found"

    if best_similarity < 0.40:
        return "ESCALATE", "Low historical similarity"

    return "AUTO-HANDLE", "High-confidence historical match"


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("SUPPORT AGENT EVALUATION")
    print("=" * 80)

    required_files = [
        INPUT_FILE,
        CLASSIFIER_FILE,
        VECTORIZER_FILE,
        RETRIEVER_FILE,
        REPLY_DATABASE_FILE
    ]

    print("\nChecking required files...")

    for file in required_files:

        if not os.path.exists(file):

            print("\nERROR: Missing file:")
            print(file)

            return

        print("FOUND:", file)

    print("\nLoading models...")

    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    retriever = joblib.load(
        RETRIEVER_FILE
    )

    reply_database = pd.read_csv(
        REPLY_DATABASE_FILE,
        low_memory=False
    )

    print(
        "Historical reply database:",
        len(reply_database)
    )

    print("\nLoading AppleSupport conversations...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

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
        "Unique customer conversations:",
        len(df)
    )

    print(
        "\nSampling",
        SAMPLE_SIZE,
        "evaluation examples..."
    )

    sample_size = min(
        SAMPLE_SIZE,
        len(df)
    )

    evaluation_df = df.sample(
        n=sample_size,
        random_state=123
    ).copy()

    print(
        "Evaluation examples:",
        len(evaluation_df)
    )

    print("\nRunning support agent...")

    results = []

    for index, row in evaluation_df.iterrows():

        customer_text = row["customer_text"]

        # Predict intent
        predicted_intent = classifier.predict(
            [customer_text]
        )[0]

        probabilities = classifier.predict_proba(
            [customer_text]
        )[0]

        confidence = probabilities.max()

        # Convert customer message into vector
        query_vector = vectorizer.transform(
            [customer_text]
        )

        # Retrieve nearest historical examples
        distances, indices = retriever.kneighbors(
            query_vector,
            n_neighbors=20
        )

        similarities = 1 - distances[0]

        retrieved_count = 0
        best_similarity = 0.0
        historical_reply = ""

        # Search for a historical example
        for similarity, database_index in zip(
            similarities,
            indices[0]
        ):

            candidate = reply_database.iloc[
                database_index
            ]

            if (
                candidate["intent"]
                == predicted_intent
            ):

                retrieved_count += 1

                if historical_reply == "":

                    historical_reply = str(
                        candidate["brand_text"]
                    )

                    best_similarity = float(
                        similarity
                    )

        action, reason = decide_action(
            customer_text,
            predicted_intent,
            confidence,
            best_similarity,
            retrieved_count
        )

        results.append({

            "customer_tweet_id":
                row["customer_tweet_id"],

            "customer_text":
                customer_text,

            "historical_brand_reply":
                row["brand_text"],

            "predicted_intent":
                predicted_intent,

            "confidence":
                round(
                    float(confidence),
                    4
                ),

            "best_similarity":
                round(
                    float(best_similarity),
                    4
                ),

            "retrieved_count":
                retrieved_count,

            "action":
                action,

            "decision_reason":
                reason,

            "retrieved_reply":
                historical_reply
        })

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(
        "\nTotal evaluated:",
        len(results_df)
    )

    print("\nAction distribution:")
    print(
        results_df["action"]
        .value_counts()
    )

    print("\nIntent distribution:")
    print(
        results_df["predicted_intent"]
        .value_counts()
    )

    print("\nAverage confidence:")
    print(
        round(
            results_df["confidence"].mean(),
            4
        )
    )

    print("\nAverage historical similarity:")
    print(
        round(
            results_df["best_similarity"].mean(),
            4
        )
    )

    print("\nEscalation reasons:")
    print(
        results_df[
            results_df["action"] == "ESCALATE"
        ]["decision_reason"]
        .value_counts()
    )

    print("\n")
    print("=" * 80)
    print("SAMPLE AGENT DECISIONS")
    print("=" * 80)

    for _, row in results_df.head(10).iterrows():

        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nPREDICTED INTENT:")
        print(row["predicted_intent"])

        print(
            "CONFIDENCE:",
            row["confidence"]
        )

        print(
            "SIMILARITY:",
            row["best_similarity"]
        )

        print(
            "ACTION:",
            row["action"]
        )

        print(
            "REASON:",
            row["decision_reason"]
        )

        print("\nRETRIEVED REPLY:")
        print(row["retrieved_reply"])

        print("-" * 80)

    print("\nEvaluation saved to:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print(
        """
This evaluation uses automatically generated
development labels and a sampled historical set.

It is useful for system development, but it is
NOT a substitute for the assignment's required
150-250 hand-labelled golden evaluation set.
"""
    )


if __name__ == "__main__":
    main()