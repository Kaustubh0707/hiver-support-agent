import os
import re
import pandas as pd
import joblib


CLASSIFIER_FILE = "models/intent_classifier.joblib"
VECTORIZER_FILE = "models/intent_reply_vectorizer.joblib"
RETRIEVER_FILE = "models/intent_reply_retriever.joblib"
DATABASE_FILE = "models/intent_reply_database.csv"

OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "reply_quality_evaluation.csv"
)

CONFIDENCE_THRESHOLD = 0.70
SIMILARITY_THRESHOLD = 0.50


def load_models():

    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    retriever = joblib.load(
        RETRIEVER_FILE
    )

    database = pd.read_csv(
        DATABASE_FILE,
        low_memory=False
    )

    database["customer_text"] = (
        database["customer_text"]
        .fillna("")
        .astype(str)
    )

    database["brand_text"] = (
        database["brand_text"]
        .fillna("")
        .astype(str)
    )

    database["intent"] = (
        database["intent"]
        .fillna("")
        .astype(str)
    )

    return (
        classifier,
        vectorizer,
        retriever,
        database
    )


def retrieve_evidence(
    customer_text,
    predicted_intent,
    vectorizer,
    retriever,
    database
):

    query_vector = vectorizer.transform(
        [customer_text]
    )

    distances, indices = retriever.kneighbors(
        query_vector,
        n_neighbors=20
    )

    candidates = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        row = database.iloc[index]

        if row["intent"] != predicted_intent:
            continue

        similarity = 1 - distance

        candidates.append({
            "customer_text": row["customer_text"],
            "brand_text": row["brand_text"],
            "similarity": similarity
        })

    candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return candidates[:3]


def contains_any(text, words):

    text = text.lower()

    for word in words:

        if word in text:
            return True

    return False


def generate_reply(intent):

    replies = {

        "battery_issue":
            "We can help with the battery issue. "
            "Please check your device's battery usage and "
            "charging behavior. If the issue continues, "
            "please contact Apple Support for further assistance.",

        "software_update_issue":
            "We can help with the software update issue. "
            "Please make sure your device has enough storage "
            "and is connected to a stable network, then try "
            "the update again. If the issue continues, "
            "Apple Support can investigate further.",

        "app_or_itunes_issue":
            "We can help with the app or iTunes issue. "
            "Please check your internet connection and make "
            "sure the relevant app or service is updated. "
            "If the issue continues, Apple Support can "
            "investigate further.",

        "connectivity_issue":
            "We can help with the connectivity issue. "
            "Please check your network connection and try "
            "reconnecting the affected service. If the problem "
            "continues, Apple Support can investigate the issue.",

        "information_or_how_to":
            "We'd be happy to help with this. "
            "Please share a little more information about "
            "what you are trying to do so the appropriate "
            "support steps can be provided."
    }

    return replies.get(
        intent,
        "We'd be happy to help with this issue. "
        "Please contact Apple Support so the issue can "
        "be investigated."
    )


def evaluate_reply(
    customer_text,
    reply,
    intent,
    confidence,
    similarity,
    action
):

    score = 0

    text = customer_text.lower()
    reply_lower = reply.lower()

    # --------------------------------------------------
    # 1. Relevance to customer issue
    # --------------------------------------------------

    relevance_words = {

        "battery_issue": [
            "battery",
            "charging",
            "charge"
        ],

        "software_update_issue": [
            "update",
            "software",
            "device"
        ],

        "app_or_itunes_issue": [
            "app",
            "itunes",
            "service"
        ],

        "connectivity_issue": [
            "network",
            "connection",
            "connecting"
        ],

        "information_or_how_to": [
            "help",
            "information",
            "share"
        ],

        "performance_freezing": [
            "issue",
            "problem",
            "investigate"
        ],

        "device_functionality_issue": [
            "issue",
            "problem",
            "investigate"
        ],

        "account_or_billing_issue": [
            "account",
            "billing",
            "investigate"
        ],

        "general_support": [
            "help",
            "issue",
            "support"
        ],

        "other": [
            "issue",
            "support"
        ]
    }

    keywords = relevance_words.get(
        intent,
        ["issue", "support"]
    )

    relevance = any(
        word in reply_lower
        for word in keywords
    )

    if relevance:
        score += 2


    # --------------------------------------------------
    # 2. Reply is not empty
    # --------------------------------------------------

    if len(reply.strip()) >= 30:
        score += 1


    # --------------------------------------------------
    # 3. Safe escalation behavior
    # --------------------------------------------------

    if action == "ESCALATE":

        if (
            "investigate" in reply_lower
            or "support" in reply_lower
            or "look into" in reply_lower
        ):
            score += 2

    else:

        score += 1


    # --------------------------------------------------
    # 4. Avoid unsupported claims
    # --------------------------------------------------

    risky_words = [
        "guarantee",
        "definitely",
        "100%",
        "will fix"
    ]

    if not contains_any(
        reply_lower,
        risky_words
    ):
        score += 1


    # --------------------------------------------------
    # 5. Appropriate automation confidence
    # --------------------------------------------------

    if action == "AUTO-HANDLE":

        if (
            confidence >= CONFIDENCE_THRESHOLD
            and similarity >= SIMILARITY_THRESHOLD
        ):
            score += 2

    else:

        score += 1


    # --------------------------------------------------
    # Maximum = 9
    # --------------------------------------------------

    return score


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("REPLY QUALITY EVALUATION")
    print("=" * 80)

    print("\nLoading models...")

    (
        classifier,
        vectorizer,
        retriever,
        database
    ) = load_models()

    print("Models loaded successfully.")

    print("\nSampling evaluation conversations...")

    evaluation_df = database.sample(
        n=min(500, len(database)),
        random_state=456
    ).copy()

    results = []

    print(
        "Evaluating",
        len(evaluation_df),
        "examples..."
    )

    for _, row in evaluation_df.iterrows():

        customer_text = row["customer_text"]

        probabilities = classifier.predict_proba(
            [customer_text]
        )[0]

        predicted_intent = classifier.predict(
            [customer_text]
        )[0]

        confidence = probabilities.max()

        evidence = retrieve_evidence(
            customer_text,
            predicted_intent,
            vectorizer,
            retriever,
            database
        )

        best_similarity = 0

        if len(evidence) > 0:
            best_similarity = evidence[0]["similarity"]

        # Decision policy
        if predicted_intent in {
            "account_or_billing_issue",
            "performance_freezing",
            "device_functionality_issue",
            "general_support",
            "other"
        }:

            action = "ESCALATE"

        elif confidence < CONFIDENCE_THRESHOLD:

            action = "ESCALATE"

        elif len(evidence) == 0:

            action = "ESCALATE"

        elif best_similarity < SIMILARITY_THRESHOLD:

            action = "ESCALATE"

        else:

            action = "AUTO-HANDLE"

        reply = generate_reply(
            predicted_intent
        )

        score = evaluate_reply(
            customer_text,
            reply,
            predicted_intent,
            confidence,
            best_similarity,
            action
        )

        results.append({

            "customer_tweet_id":
                row.get(
                    "customer_tweet_id",
                    ""
                ),

            "customer_text":
                customer_text,

            "predicted_intent":
                predicted_intent,

            "confidence":
                confidence,

            "evidence_count":
                len(evidence),

            "best_similarity":
                best_similarity,

            "action":
                action,

            "reply":
                reply,

            "quality_score":
                score
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
    print("REPLY QUALITY RESULTS")
    print("=" * 80)

    print(
        "\nExamples evaluated:",
        len(results_df)
    )

    print(
        "\nAverage quality score:",
        round(
            results_df["quality_score"].mean(),
            4
        ),
        "/ 9"
    )

    print(
        "\nMedian quality score:",
        round(
            results_df["quality_score"].median(),
            4
        ),
        "/ 9"
    )

    print("\nQuality score distribution:")
    print("-" * 60)

    print(
        results_df[
            "quality_score"
        ].value_counts().sort_index()
    )

    print("\nAction distribution:")
    print("-" * 60)

    print(
        results_df[
            "action"
        ].value_counts()
    )

    print("\nAverage quality by action:")
    print("-" * 60)

    print(
        results_df.groupby(
            "action"
        )["quality_score"].mean()
    )

    print("\nHigh-quality replies (score >= 7):")

    high_quality = (
        results_df["quality_score"] >= 7
    ).mean()

    print(
        round(
            high_quality * 100,
            2
        ),
        "%"
    )

    print("\nSaved evaluation file:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print(
        "This is a local rule-based proxy evaluation."
    )

    print(
        "It is NOT an LLM-as-judge evaluation."
    )

    print(
        "Human/LLM agreement has not been measured."
    )

    print("\nReply quality evaluation completed.")


if __name__ == "__main__":
    main()