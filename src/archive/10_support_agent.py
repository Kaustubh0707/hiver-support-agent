import os
import joblib
import pandas as pd


CLASSIFIER_FILE = "models/intent_classifier.joblib"
VECTORIZER_FILE = "models/reply_vectorizer.joblib"
RETRIEVER_FILE = "models/reply_retriever.joblib"
REPLY_DATABASE_FILE = "models/reply_database.csv"


def load_models():

    print("Loading models...")

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

    print("Models loaded successfully.")

    return (
        classifier,
        vectorizer,
        retriever,
        reply_database
    )


def decide_action(text, intent, confidence):

    text_lower = text.lower()

    escalation_words = [
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
        word in text_lower
        for word in escalation_words
    ):

        return (
            "ESCALATE",
            "The message may require account-specific "
            "or troubleshooting investigation."
        )


    if intent in [
        "general_support",
        "other",
        "performance_freezing",
        "device_functionality_issue"
    ]:

        return (
            "ESCALATE",
            "The issue is unclear or may require "
            "device-specific troubleshooting."
        )


    if confidence < 0.60:

        return (
            "ESCALATE",
            "The intent classifier has low confidence."
        )


    return (
        "AUTO-HANDLE",
        "The issue matches a known support intent "
        "with sufficient confidence."
    )


def clean_reply(reply):

    reply = str(reply)

    # Remove obvious Twitter usernames from beginning.
    parts = reply.split(" ", 1)

    if len(parts) == 2 and parts[0].startswith("@"):
        reply = parts[1]

    return reply.strip()


def run_agent(
    message,
    classifier,
    vectorizer,
    retriever,
    reply_database
):

    # ---------------------------------------------------------
    # Intent classification
    # ---------------------------------------------------------

    intent = classifier.predict(
        [message]
    )[0]

    probabilities = classifier.predict_proba(
        [message]
    )[0]

    confidence = probabilities.max()


    # ---------------------------------------------------------
    # Historical retrieval
    # ---------------------------------------------------------

    query_vector = vectorizer.transform(
        [message]
    )

    distances, indices = retriever.kneighbors(
        query_vector,
        n_neighbors=3
    )


    historical_cases = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        similarity = 1 - distance

        row = reply_database.iloc[index]

        historical_cases.append({
            "similarity": similarity,
            "customer_text": row["customer_text"],
            "brand_text": row["brand_text"]
        })


    # ---------------------------------------------------------
    # Action decision
    # ---------------------------------------------------------

    action, reason = decide_action(
        message,
        intent,
        confidence
    )


    # ---------------------------------------------------------
    # Select grounded reply
    # ---------------------------------------------------------

    best_reply = clean_reply(
        historical_cases[0]["brand_text"]
    )


    return {
        "intent": intent,
        "confidence": confidence,
        "action": action,
        "reason": reason,
        "reply": best_reply,
        "historical_cases": historical_cases
    }


def print_result(message, result):

    print("\n")
    print("=" * 80)
    print("SUPPORT AGENT RESULT")
    print("=" * 80)

    print("\nCUSTOMER MESSAGE:")
    print(message)

    print("\nPREDICTED INTENT:")
    print(result["intent"])

    print("\nCONFIDENCE:")
    print(round(result["confidence"], 4))

    print("\nDECISION:")
    print(result["action"])

    print("\nREASON:")
    print(result["reason"])

    print("\nDRAFT REPLY:")
    print(result["reply"])

    print("\n" + "-" * 80)
    print("HISTORICAL EVIDENCE")
    print("-" * 80)

    for number, case in enumerate(
        result["historical_cases"],
        start=1
    ):

        print("\nCASE", number)

        print(
            "Similarity:",
            round(case["similarity"], 4)
        )

        print(
            "Customer:",
            case["customer_text"]
        )

        print(
            "AppleSupport:",
            case["brand_text"]
        )


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("END-TO-END SUPPORT AGENT")
    print("=" * 80)

    required_files = [
        CLASSIFIER_FILE,
        VECTORIZER_FILE,
        RETRIEVER_FILE,
        REPLY_DATABASE_FILE
    ]

    for file in required_files:

        if not os.path.exists(file):

            print("\nERROR: Required file not found:")
            print(file)

            return


    (
        classifier,
        vectorizer,
        retriever,
        reply_database
    ) = load_models()


    test_messages = [

        "My iPhone battery is draining very quickly.",

        "My iPhone keeps restarting and I cannot use it.",

        "Bluetooth is not connecting to my headphones.",

        "I cannot download anything from the App Store.",

        "How do I update my iPhone?"
    ]


    for message in test_messages:

        result = run_agent(
            message,
            classifier,
            vectorizer,
            retriever,
            reply_database
        )

        print_result(
            message,
            result
        )


if __name__ == "__main__":
    main()