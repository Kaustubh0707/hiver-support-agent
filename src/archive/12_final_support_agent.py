import os
import joblib
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

CLASSIFIER_FILE = "models/intent_classifier.joblib"

VECTORIZER_FILE = "models/intent_reply_vectorizer.joblib"

RETRIEVER_FILE = "models/intent_reply_retriever.joblib"

DATABASE_FILE = "models/intent_reply_database.csv"


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 3

# Minimum similarity required before using historical evidence.
# If similarity is below this value, we escalate instead.
SIMILARITY_THRESHOLD = 0.40

# Minimum classifier confidence.
CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("Loading support agent models...")

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

    print("All models loaded successfully.")

    return (
        classifier,
        vectorizer,
        retriever,
        database
    )


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def predict_intent(
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


# ============================================================
# RETRIEVE HISTORICAL CASES
# ============================================================

def retrieve_cases(
    message,
    predicted_intent,
    vectorizer,
    retriever,
    database
):

    query_vector = vectorizer.transform(
        [message]
    )

    distances, indices = retriever.kneighbors(
        query_vector,
        n_neighbors=20
    )

    matching_cases = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        similarity = 1 - distance

        row = database.iloc[index]

        # Only use historical conversations
        # belonging to the predicted intent.
        if row["intent"] != predicted_intent:
            continue

        matching_cases.append(
            {
                "similarity": similarity,
                "customer_text": row["customer_text"],
                "brand_text": row["brand_text"],
                "intent": row["intent"]
            }
        )

    # Sort from most similar to least similar.

    matching_cases.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return matching_cases[:TOP_K]


# ============================================================
# CHECK FOR SENSITIVE / HIGH-RISK CASES
# ============================================================

def contains_escalation_signal(
    message
):

    text = message.lower()

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

    for word in escalation_words:

        if word in text:
            return True

    return False


# ============================================================
# DECIDE AUTO-HANDLE OR ESCALATE
# ============================================================

def decide_action(
    message,
    intent,
    confidence,
    historical_cases
):

    # --------------------------------------------------------
    # Rule 1: Sensitive/account-specific issues
    # --------------------------------------------------------

    if contains_escalation_signal(
        message
    ):

        return (
            "ESCALATE",
            "The message may require account-specific "
            "or troubleshooting investigation."
        )


    # --------------------------------------------------------
    # Rule 2: Unclear intents
    # --------------------------------------------------------

    if intent in [
        "general_support",
        "other"
    ]:

        return (
            "ESCALATE",
            "The customer request is unclear "
            "or does not match a specific support intent."
        )


    # --------------------------------------------------------
    # Rule 3: Device-specific troubleshooting
    # --------------------------------------------------------

    if intent in [
        "performance_freezing",
        "device_functionality_issue"
    ]:

        return (
            "ESCALATE",
            "The issue may require device-specific "
            "troubleshooting."
        )


    # --------------------------------------------------------
    # Rule 4: Low classifier confidence
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        return (
            "ESCALATE",
            "The intent classifier has low confidence."
        )


    # --------------------------------------------------------
    # Rule 5: No historical evidence
    # --------------------------------------------------------

    if len(historical_cases) == 0:

        return (
            "ESCALATE",
            "No historical support example was found "
            "for the predicted intent."
        )


    # --------------------------------------------------------
    # Rule 6: Weak historical evidence
    # --------------------------------------------------------

    best_similarity = historical_cases[0]["similarity"]

    if best_similarity < SIMILARITY_THRESHOLD:

        return (
            "ESCALATE",
            "The closest historical support example "
            "is not sufficiently similar."
        )


    # --------------------------------------------------------
    # Otherwise auto-handle
    # --------------------------------------------------------

    return (
        "AUTO-HANDLE",
        "The intent is confident and a sufficiently "
        "similar historical support example was found."
    )


# ============================================================
# CLEAN HISTORICAL REPLY
# ============================================================

def clean_reply(
    reply
):

    reply = str(reply).strip()

    # Remove Twitter username at the beginning.

    parts = reply.split(
        " ",
        1
    )

    if (
        len(parts) == 2
        and parts[0].startswith("@")
    ):

        reply = parts[1]

    return reply.strip()


# ============================================================
# RUN COMPLETE AGENT
# ============================================================

def run_agent(
    message,
    classifier,
    vectorizer,
    retriever,
    database
):

    # --------------------------------------------------------
    # Step 1: Intent
    # --------------------------------------------------------

    intent, confidence = predict_intent(
        message,
        classifier
    )


    # --------------------------------------------------------
    # Step 2: Historical retrieval
    # --------------------------------------------------------

    historical_cases = retrieve_cases(
        message,
        intent,
        vectorizer,
        retriever,
        database
    )


    # --------------------------------------------------------
    # Step 3: Decision
    # --------------------------------------------------------

    action, reason = decide_action(
        message,
        intent,
        confidence,
        historical_cases
    )


    # --------------------------------------------------------
    # Step 4: Select reply
    # --------------------------------------------------------

    if (
        action == "AUTO-HANDLE"
        and len(historical_cases) > 0
    ):

        reply = clean_reply(
            historical_cases[0]["brand_text"]
        )

    else:

        reply = (
            "This issue needs further investigation. "
            "A support specialist should review the case."
        )


    return {

        "intent": intent,

        "confidence": confidence,

        "action": action,

        "reason": reason,

        "reply": reply,

        "historical_cases": historical_cases
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    message,
    result
):

    print("\n")
    print("=" * 80)

    print(
        "CUSTOMER MESSAGE"
    )

    print("=" * 80)

    print(message)


    print("\n")
    print(
        "PREDICTED INTENT:"
    )

    print(
        result["intent"]
    )


    print("\n")
    print(
        "CLASSIFIER CONFIDENCE:"
    )

    print(
        round(
            result["confidence"],
            4
        )
    )


    print("\n")
    print(
        "DECISION:"
    )

    print(
        result["action"]
    )


    print("\n")
    print(
        "REASON:"
    )

    print(
        result["reason"]
    )


    print("\n")
    print(
        "DRAFT REPLY:"
    )

    print(
        result["reply"]
    )


    print("\n")
    print(
        "HISTORICAL EVIDENCE:"
    )

    print("-" * 80)


    if len(
        result["historical_cases"]
    ) == 0:

        print(
            "No matching historical cases."
        )

        return


    for number, case in enumerate(
        result["historical_cases"],
        start=1
    ):

        print("\nCASE", number)

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
            "Customer:",
            case["customer_text"]
        )

        print(
            "AppleSupport:",
            case["brand_text"]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "HIVER AI SUPPORT AGENT"
    )

    print(
        "FINAL END-TO-END AGENT"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [

        CLASSIFIER_FILE,

        VECTORIZER_FILE,

        RETRIEVER_FILE,

        DATABASE_FILE
    ]


    for file in required_files:

        if not os.path.exists(file):

            print("\nERROR: Required file not found:")

            print(file)

            return


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    (
        classifier,
        vectorizer,
        retriever,
        database
    ) = load_models()


    # --------------------------------------------------------
    # Test customer messages
    # --------------------------------------------------------

    test_messages = [

        "My iPhone battery is draining very quickly.",

        "My iPhone keeps restarting and I cannot use it.",

        "Bluetooth is not connecting to my headphones.",

        "I cannot download anything from the App Store.",

        "How do I update my iPhone?",

        "I was charged for something I did not purchase.",

        "My screen is not responding.",

        "Can you help me with my iPhone?"
    ]


    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    for message in test_messages:

        result = run_agent(

            message,

            classifier,

            vectorizer,

            retriever,

            database
        )

        print_result(
            message,
            result
        )


if __name__ == "__main__":

    main()