import os
import re
import pandas as pd
import joblib


CLASSIFIER_FILE = "models/intent_classifier.joblib"
VECTORIZER_FILE = "models/intent_reply_vectorizer.joblib"
RETRIEVER_FILE = "models/intent_reply_retriever.joblib"
DATABASE_FILE = "models/intent_reply_database.csv"


CONFIDENCE_THRESHOLD = 0.70
SIMILARITY_THRESHOLD = 0.50


HIGH_RISK_INTENTS = {
    "general_support",
    "other",
    "performance_freezing",
    "device_functionality_issue",
    "account_or_billing_issue"
}


ESCALATION_SIGNALS = [
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
    "nothing works",
    "completely dead"
]


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_models():

    print("\nLoading models...")

    if not os.path.exists(CLASSIFIER_FILE):
        raise FileNotFoundError(CLASSIFIER_FILE)

    if not os.path.exists(VECTORIZER_FILE):
        raise FileNotFoundError(VECTORIZER_FILE)

    if not os.path.exists(RETRIEVER_FILE):
        raise FileNotFoundError(RETRIEVER_FILE)

    if not os.path.exists(DATABASE_FILE):
        raise FileNotFoundError(DATABASE_FILE)

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

    print("Models loaded successfully.")

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
        key=lambda x: x["similarity"],
        reverse=True
    )

    return candidates[:3]


def contains_escalation_signal(text):

    text = text.lower()

    for signal in ESCALATION_SIGNALS:

        if signal in text:
            return True

    return False


def decide_action(
    customer_text,
    predicted_intent,
    confidence,
    evidence
):

    text = customer_text.lower()

    # ---------------------------------------------------------
    # 1. Sensitive or risky cases
    # ---------------------------------------------------------

    if contains_escalation_signal(text):

        return (
            "ESCALATE",
            "Sensitive or unresolved issue"
        )

    # ---------------------------------------------------------
    # 2. Broad / unclear intents
    # ---------------------------------------------------------

    if predicted_intent in HIGH_RISK_INTENTS:

        if predicted_intent == "account_or_billing_issue":
            return (
                "ESCALATE",
                "Account or billing issue requires investigation"
            )

        if predicted_intent == "performance_freezing":
            return (
                "ESCALATE",
                "Device-specific troubleshooting required"
            )

        if predicted_intent == "device_functionality_issue":
            return (
                "ESCALATE",
                "Device-specific troubleshooting required"
            )

        return (
            "ESCALATE",
            "Intent is too broad or unclear"
        )

    # ---------------------------------------------------------
    # 3. Low classifier confidence
    # ---------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        return (
            "ESCALATE",
            "Low intent confidence"
        )

    # ---------------------------------------------------------
    # 4. No historical evidence
    # ---------------------------------------------------------

    if len(evidence) == 0:

        return (
            "ESCALATE",
            "No sufficiently relevant historical evidence"
        )

    # ---------------------------------------------------------
    # 5. Weak historical evidence
    # ---------------------------------------------------------

    best_similarity = evidence[0]["similarity"]

    if best_similarity < SIMILARITY_THRESHOLD:

        return (
            "ESCALATE",
            "Low historical similarity"
        )

    # ---------------------------------------------------------
    # 6. Safe to auto-handle
    # ---------------------------------------------------------

    return (
        "AUTO-HANDLE",
        "High-confidence intent with sufficient historical evidence"
    )


def generate_local_reply(
    customer_text,
    predicted_intent,
    evidence
):

    # ---------------------------------------------------------
    # IMPORTANT:
    # Do not directly expose the historical customer message
    # or copy a complete historical reply.
    # ---------------------------------------------------------

    reply_templates = {

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
            "Please check that your device is connected to "
            "the internet and that the relevant app or service "
            "is updated. If the issue continues, please contact "
            "Apple Support for further assistance.",

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

    if predicted_intent in reply_templates:

        return reply_templates[predicted_intent]

    return (
        "We'd be happy to help with this issue. "
        "Please contact Apple Support so the problem can "
        "be investigated and the appropriate assistance "
        "can be provided."
    )


def run_agent(
    customer_text,
    classifier,
    vectorizer,
    retriever,
    database
):

    # ---------------------------------------------------------
    # Intent classification
    # ---------------------------------------------------------

    probabilities = classifier.predict_proba(
        [customer_text]
    )[0]

    predicted_intent = classifier.predict(
        [customer_text]
    )[0]

    confidence = probabilities.max()

    # ---------------------------------------------------------
    # Historical retrieval
    # ---------------------------------------------------------

    evidence = retrieve_evidence(
        customer_text,
        predicted_intent,
        vectorizer,
        retriever,
        database
    )

    # ---------------------------------------------------------
    # Decision
    # ---------------------------------------------------------

    action, reason = decide_action(
        customer_text,
        predicted_intent,
        confidence,
        evidence
    )

    # ---------------------------------------------------------
    # Reply
    # ---------------------------------------------------------

    if action == "AUTO-HANDLE":

        reply = generate_local_reply(
            customer_text,
            predicted_intent,
            evidence
        )

    else:

        reply = (
            "Thanks for reaching out. "
            "We'd like to look into this further. "
            "Please contact Apple Support so the issue "
            "can be investigated."
        )

    best_similarity = 0

    if len(evidence) > 0:

        best_similarity = evidence[0]["similarity"]

    return {
        "customer_text": customer_text,
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "action": action,
        "decision_reason": reason,
        "best_similarity": best_similarity,
        "evidence_count": len(evidence),
        "reply": reply
    }


def print_result(result):

    print("\n")
    print("=" * 80)
    print("CUSTOMER MESSAGE")
    print("=" * 80)

    print(result["customer_text"])

    print("\nPredicted intent:")
    print(result["predicted_intent"])

    print("\nConfidence:")
    print(round(result["confidence"], 4))

    print("\nHistorical evidence:")
    print(result["evidence_count"])

    print("\nBest historical similarity:")
    print(round(result["best_similarity"], 4))

    print("\nDecision:")
    print(result["action"])

    print("\nReason:")
    print(result["decision_reason"])

    print("\nGenerated reply:")
    print(result["reply"])


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("FINAL SUPPORT AGENT V2")
    print("=" * 80)

    (
        classifier,
        vectorizer,
        retriever,
        database
    ) = load_models()

    sample_messages = [

        "My iPhone battery is draining very quickly.",

        "My phone keeps restarting after the update.",

        "Bluetooth is not connecting.",

        "iTunes is not working.",

        "How do I update my iPhone?",

        "I was charged for something I did not purchase.",

        "My screen is not responding.",

        "Can you help me with my iPhone?"
    ]

    print("\n")
    print("=" * 80)
    print("TESTING FINAL AGENT")
    print("=" * 80)

    for message in sample_messages:

        result = run_agent(
            message,
            classifier,
            vectorizer,
            retriever,
            database
        )

        print_result(result)

    print("\n")
    print("=" * 80)
    print("FINAL AGENT TEST COMPLETED")
    print("=" * 80)

    print("\nPolicy used:")

    print(
        f"- Intent confidence >= "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"- Historical similarity >= "
        f"{SIMILARITY_THRESHOLD}"
    )

    print(
        "- High-risk/device-specific issues escalate"
    )

    print(
        "- Sensitive account/billing issues escalate"
    )


if __name__ == "__main__":
    main()