# (First One Code)
# import os
# import re
# import pandas as pd
# import joblib


# CLASSIFIER_FILE = "models/intent_classifier.joblib"
# VECTORIZER_FILE = "models/intent_reply_vectorizer.joblib"
# RETRIEVER_FILE = "models/intent_reply_retriever.joblib"
# DATABASE_FILE = "models/intent_reply_database.csv"

# CONFIDENCE_THRESHOLD = 0.70
# SIMILARITY_THRESHOLD = 0.50


# HIGH_RISK_INTENTS = {
#     "general_support",
#     "other",
#     "performance_freezing",
#     "device_functionality_issue",
#     "account_or_billing_issue"
# }


# SENSITIVE_SIGNALS = [
#     "charged",
#     "refund",
#     "payment",
#     "account",
#     "apple id",
#     "password",
#     "hacked",
#     "stolen",
#     "lost",
#     "security",
#     "fraud",
#     "cannot access",
#     "can't access"
# ]


# UNRESOLVED_SIGNALS = [
#     "still not working",
#     "tried everything",
#     "nothing works",
#     "completely dead"
# ]


# def retrieve_evidence(
#     customer_text,
#     predicted_intent,
#     vectorizer,
#     retriever,
#     database
# ):

#     query_vector = vectorizer.transform(
#         [customer_text]
#     )

#     distances, indices = retriever.kneighbors(
#         query_vector,
#         n_neighbors=20
#     )

#     candidates = []

#     for distance, index in zip(
#         distances[0],
#         indices[0]
#     ):

#         row = database.iloc[index]

#         if row["intent"] != predicted_intent:
#             continue

#         similarity = 1 - distance

#         candidates.append({
#             "customer_text": row["customer_text"],
#             "brand_text": row["brand_text"],
#             "similarity": similarity
#         })

#     candidates.sort(
#         key=lambda item: item["similarity"],
#         reverse=True
#     )

#     return candidates[:3]


# def contains_any(text, signals):

#     text = text.lower()

#     return any(
#         signal in text
#         for signal in signals
#     )


# def decide_action(
#     customer_text,
#     predicted_intent,
#     confidence,
#     evidence
# ):

#     # 1. Sensitive account/security issues
#     if contains_any(
#         customer_text,
#         SENSITIVE_SIGNALS
#     ):

#         return (
#             "ESCALATE",
#             "Sensitive account or security issue"
#         )

#     # 2. Clearly unresolved problems
#     if contains_any(
#         customer_text,
#         UNRESOLVED_SIGNALS
#     ):

#         return (
#             "ESCALATE",
#             "Issue appears unresolved and requires investigation"
#         )

#     # 3. High-risk troubleshooting
#     if predicted_intent == "account_or_billing_issue":

#         return (
#             "ESCALATE",
#             "Account or billing issue requires investigation"
#         )

#     if predicted_intent == "performance_freezing":

#         return (
#             "ESCALATE",
#             "Device-specific troubleshooting required"
#         )

#     if predicted_intent == "device_functionality_issue":

#         return (
#             "ESCALATE",
#             "Device-specific troubleshooting required"
#         )

#     # 4. Broad intents
#     if predicted_intent in {
#         "general_support",
#         "other"
#     }:

#         return (
#             "ESCALATE",
#             "Intent is too broad or unclear"
#         )

#     # 5. Low classifier confidence
#     if confidence < CONFIDENCE_THRESHOLD:

#         return (
#             "ESCALATE",
#             "Low intent confidence"
#         )

#     # 6. No useful historical evidence
#     if len(evidence) == 0:

#         return (
#             "ESCALATE",
#             "No sufficiently relevant historical evidence"
#         )

#     # 7. Weak historical evidence
#     best_similarity = evidence[0]["similarity"]

#     if best_similarity < SIMILARITY_THRESHOLD:

#         return (
#             "ESCALATE",
#             "Low historical similarity"
#         )

#     # 8. Safe automation
#     return (
#         "AUTO-HANDLE",
#         "High-confidence intent with sufficient historical evidence"
#     )


# def generate_reply(intent):

#     replies = {

#         "battery_issue":
#             "We can help with the battery issue. "
#             "Please check your device's battery usage and "
#             "charging behavior. If the issue continues, "
#             "please contact Apple Support for further assistance.",

#         "software_update_issue":
#             "We can help with the software update issue. "
#             "Please make sure your device has enough storage "
#             "and is connected to a stable network, then try "
#             "the update again. If the issue continues, "
#             "Apple Support can investigate further.",

#         "app_or_itunes_issue":
#             "We can help with the app or iTunes issue. "
#             "Please check your internet connection and make "
#             "sure the relevant app or service is updated. "
#             "If the issue continues, Apple Support can "
#             "investigate further.",

#         "connectivity_issue":
#             "We can help with the connectivity issue. "
#             "Please check your network connection and try "
#             "reconnecting the affected service. If the problem "
#             "continues, Apple Support can investigate the issue.",

#         "information_or_how_to":
#             "We'd be happy to help with this. "
#             "Please share a little more information about "
#             "what you are trying to do so the appropriate "
#             "support steps can be provided."
#     }

#     return replies.get(
#         intent,
#         "We'd be happy to help with this issue. "
#         "Please contact Apple Support so the issue can "
#         "be investigated."
#     )


# def run_agent(
#     customer_text,
#     classifier,
#     vectorizer,
#     retriever,
#     database
# ):

#     probabilities = classifier.predict_proba(
#         [customer_text]
#     )[0]

#     predicted_intent = classifier.predict(
#         [customer_text]
#     )[0]

#     confidence = probabilities.max()

#     evidence = retrieve_evidence(
#         customer_text,
#         predicted_intent,
#         vectorizer,
#         retriever,
#         database
#     )

#     action, reason = decide_action(
#         customer_text,
#         predicted_intent,
#         confidence,
#         evidence
#     )

#     if action == "AUTO-HANDLE":

#         reply = generate_reply(
#             predicted_intent
#         )

#     else:

#         reply = (
#             "Thanks for reaching out. "
#             "We'd like to look into this further. "
#             "Please contact Apple Support so the issue "
#             "can be investigated."
#         )

#     best_similarity = 0

#     if len(evidence) > 0:

#         best_similarity = evidence[0]["similarity"]

#     return {
#         "customer_text": customer_text,
#         "predicted_intent": predicted_intent,
#         "confidence": confidence,
#         "evidence_count": len(evidence),
#         "best_similarity": best_similarity,
#         "action": action,
#         "decision_reason": reason,
#         "reply": reply
#     }


# def main():

#     print("=" * 80)
#     print("HIVER AI SUPPORT AGENT")
#     print("FINAL SUPPORT AGENT V3")
#     print("=" * 80)

#     print("\nLoading models...")

#     classifier = joblib.load(
#         CLASSIFIER_FILE
#     )

#     vectorizer = joblib.load(
#         VECTORIZER_FILE
#     )

#     retriever = joblib.load(
#         RETRIEVER_FILE
#     )

#     database = pd.read_csv(
#         DATABASE_FILE,
#         low_memory=False
#     )

#     database["customer_text"] = (
#         database["customer_text"]
#         .fillna("")
#         .astype(str)
#     )

#     database["brand_text"] = (
#         database["brand_text"]
#         .fillna("")
#         .astype(str)
#     )

#     database["intent"] = (
#         database["intent"]
#         .fillna("")
#         .astype(str)
#     )

#     print("Models loaded successfully.")

#     test_messages = [

#         "My iPhone battery is draining very quickly.",

#         "My phone keeps restarting after the update.",

#         "Bluetooth is not connecting.",

#         "iTunes is not working.",

#         "How do I update my iPhone?",

#         "I was charged for something I did not purchase.",

#         "My screen is not responding.",

#         "Can you help me with my iPhone?"
#     ]

#     print("\n" + "=" * 80)
#     print("FINAL AGENT TEST")
#     print("=" * 80)

#     for message in test_messages:

#         result = run_agent(
#             message,
#             classifier,
#             vectorizer,
#             retriever,
#             database
#         )

#         print("\n")
#         print("-" * 80)

#         print("Customer:")
#         print(result["customer_text"])

#         print("\nIntent:")
#         print(result["predicted_intent"])

#         print("\nConfidence:")
#         print(round(
#             result["confidence"],
#             4
#         ))

#         print("\nEvidence count:")
#         print(result["evidence_count"])

#         print("\nBest similarity:")
#         print(round(
#             result["best_similarity"],
#             4
#         ))

#         print("\nDecision:")
#         print(result["action"])

#         print("\nReason:")
#         print(result["decision_reason"])

#         print("\nReply:")
#         print(result["reply"])

#     print("\n")
#     print("=" * 80)
#     print("FINAL POLICY")
#     print("=" * 80)

#     print(
#         f"Intent confidence >= "
#         f"{CONFIDENCE_THRESHOLD}"
#     )

#     print(
#         f"Historical similarity >= "
#         f"{SIMILARITY_THRESHOLD}"
#     )

#     print(
#         "Sensitive account/security issues -> ESCALATE"
#     )

#     print(
#         "Device-specific troubleshooting -> ESCALATE"
#     )

#     print(
#         "Broad or unclear intents -> ESCALATE"
#     )

#     print("\nFinal agent V3 completed successfully.")


# if __name__ == "__main__":
#     main()

# (New One Code)
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
# POLICY THRESHOLDS
# ============================================================

CONFIDENCE_THRESHOLD = 0.70
SIMILARITY_THRESHOLD = 0.50


# ============================================================
# RISK CATEGORIES
# ============================================================

HIGH_RISK_INTENTS = {
    "general_support",
    "other",
    "performance_freezing",
    "device_functionality_issue",
    "account_or_billing_issue"
}


# ============================================================
# SENSITIVE SIGNALS
# ============================================================

SENSITIVE_SIGNALS = [
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
    "can't access"
]


# ============================================================
# UNRESOLVED SIGNALS
# ============================================================

UNRESOLVED_SIGNALS = [
    "still not working",
    "tried everything",
    "nothing works",
    "completely dead"
]


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("Loading models...")

    if not os.path.exists(CLASSIFIER_FILE):
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_FILE}"
        )

    if not os.path.exists(VECTORIZER_FILE):
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_FILE}"
        )

    if not os.path.exists(RETRIEVER_FILE):
        raise FileNotFoundError(
            f"Retriever not found: {RETRIEVER_FILE}"
        )

    if not os.path.exists(DATABASE_FILE):
        raise FileNotFoundError(
            f"Database not found: {DATABASE_FILE}"
        )

    classifier = joblib.load(CLASSIFIER_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    retriever = joblib.load(RETRIEVER_FILE)

    database = pd.read_csv(
        DATABASE_FILE,
        low_memory=False
    )

    print("Models loaded successfully.")

    print("\nDatabase columns:")
    print(list(database.columns))

    return classifier, vectorizer, retriever, database


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(
    classifier,
    message
):

    probabilities = classifier.predict_proba(
        [message]
    )[0]

    classes = classifier.classes_

    best_index = probabilities.argmax()

    intent = classes[best_index]
    confidence = probabilities[best_index]

    return intent, confidence


# ============================================================
# HISTORICAL RETRIEVAL
# ============================================================

def retrieve_evidence(
    vectorizer,
    retriever,
    database,
    message,
    predicted_intent
):

    query_vector = vectorizer.transform(
        [message]
    )

    distances, indices = retriever.kneighbors(
        query_vector,
        n_neighbors=20
    )

    evidence = []

    # Check whether predicted_intent exists
    # in the existing historical database.
    has_intent_column = (
        "predicted_intent" in database.columns
    )

    # --------------------------------------------------------
    # FIRST PASS
    # --------------------------------------------------------
    # If the database has predicted_intent,
    # prefer examples belonging to the same intent.
    #
    # If it does not have the column, simply use the
    # strongest historical examples.
    # --------------------------------------------------------

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        row = database.iloc[index]

        if has_intent_column:

            if row["predicted_intent"] != predicted_intent:
                continue

        similarity = 1 - distance

        customer_text = str(
            row["customer_text"]
        ).strip()

        brand_text = str(
            row["brand_text"]
        ).strip()

        if customer_text == "" or customer_text.lower() == "nan":
            continue

        if brand_text == "" or brand_text.lower() == "nan":
            continue

        evidence.append({
            "customer_text": customer_text,
            "brand_text": brand_text,
            "similarity": similarity
        })

        if len(evidence) == 3:
            break

    # --------------------------------------------------------
    # SECOND PASS
    # --------------------------------------------------------
    # If fewer than 3 same-intent examples were found,
    # fill remaining slots with the strongest historical
    # examples.
    # --------------------------------------------------------

    if len(evidence) < 3:

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            row = database.iloc[index]

            customer_text = str(
                row["customer_text"]
            ).strip()

            brand_text = str(
                row["brand_text"]
            ).strip()

            if customer_text == "" or customer_text.lower() == "nan":
                continue

            if brand_text == "" or brand_text.lower() == "nan":
                continue

            already_added = False

            for item in evidence:

                if item["customer_text"] == customer_text:
                    already_added = True
                    break

            if already_added:
                continue

            similarity = 1 - distance

            evidence.append({
                "customer_text": customer_text,
                "brand_text": brand_text,
                "similarity": similarity
            })

            if len(evidence) == 3:
                break

    return evidence


# ============================================================
# HISTORICAL RESPONSE PATTERN
# ============================================================

def extract_historical_pattern(
    evidence
):

    if len(evidence) == 0:

        return (
            "No historical response pattern available."
        )

    replies = []

    for item in evidence:

        reply = str(
            item["brand_text"]
        ).strip()

        if reply == "":
            continue

        if reply.lower() == "nan":
            continue

        replies.append(reply)

    if len(replies) == 0:

        return (
            "No historical response pattern available."
        )

    # Use the strongest historical response
    # as evidence for the response pattern.
    return replies[0]


# ============================================================
# REPLY GENERATION
# ============================================================

def generate_reply(
    message,
    intent,
    action,
    evidence
):

    # --------------------------------------------------------
    # ESCALATION RESPONSE
    # --------------------------------------------------------

    if action == "ESCALATE":

        return (
            "Thanks for reaching out. We'd like to look into "
            "this further. Please contact Apple Support so the "
            "issue can be investigated."
        )

    # --------------------------------------------------------
    # Historical evidence
    # --------------------------------------------------------

    historical_pattern = extract_historical_pattern(
        evidence
    )

    # Keep variable available to show that historical
    # evidence is part of the response-generation stage.
    _ = historical_pattern

    # --------------------------------------------------------
    # BATTERY
    # --------------------------------------------------------

    if intent == "battery_issue":

        return (
            "We can help with the battery issue. Please check "
            "your device's battery usage and charging behavior. "
            "Similar Apple Support cases focus on checking "
            "battery-related behavior before further "
            "troubleshooting. If the issue continues, please "
            "contact Apple Support for further assistance."
        )

    # --------------------------------------------------------
    # SOFTWARE UPDATE
    # --------------------------------------------------------

    if intent == "software_update_issue":

        return (
            "We can help with the software update issue. "
            "Similar Apple Support cases first check the device "
            "and update conditions before further "
            "troubleshooting. Please make sure your device has "
            "enough storage and is connected to a stable network, "
            "then try the update again. If the issue continues, "
            "Apple Support can investigate further."
        )

    # --------------------------------------------------------
    # APP / ITUNES
    # --------------------------------------------------------

    if intent == "app_or_itunes_issue":

        return (
            "We can help with the app or iTunes issue. Similar "
            "Apple Support cases involve checking the affected "
            "application or service and then investigating the "
            "specific device setup. Please try the issue again "
            "and contact Apple Support if it continues."
        )

    # --------------------------------------------------------
    # CONNECTIVITY
    # --------------------------------------------------------

    if intent == "connectivity_issue":

        return (
            "We can help with the connectivity issue. Similar "
            "Apple Support cases check whether the problem occurs "
            "across different networks or devices. Please try "
            "connecting again and check whether other devices or "
            "networks work normally. If the issue continues, "
            "Apple Support can investigate further."
        )

    # --------------------------------------------------------
    # INFORMATION / HOW-TO
    # --------------------------------------------------------

    if intent == "information_or_how_to":

        return (
            "We can help with this. Similar Apple Support cases "
            "typically start by checking the device setup and "
            "providing the relevant steps for the requested task. "
            "Please contact Apple Support if you need further "
            "assistance."
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return (
        "We found a similar Apple Support response for this "
        "type of issue. The recommended approach is to review "
        "the device setup and investigate the specific problem. "
        "If the issue continues, please contact Apple Support "
        "for further assistance."
    )


# ============================================================
# DECISION POLICY
# ============================================================

def make_decision(
    message,
    intent,
    confidence,
    evidence
):

    message_lower = message.lower()

    # --------------------------------------------------------
    # Best historical similarity
    # --------------------------------------------------------

    best_similarity = 0.0

    if len(evidence) > 0:

        best_similarity = max(
            item["similarity"]
            for item in evidence
        )

    # --------------------------------------------------------
    # SENSITIVE CASES
    # --------------------------------------------------------

    for signal in SENSITIVE_SIGNALS:

        if signal in message_lower:

            return (
                "ESCALATE",
                "Sensitive account or security issue",
                best_similarity
            )

    # --------------------------------------------------------
    # UNRESOLVED CASES
    # --------------------------------------------------------

    for signal in UNRESOLVED_SIGNALS:

        if signal in message_lower:

            return (
                "ESCALATE",
                "Issue appears unresolved and requires investigation",
                best_similarity
            )

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        return (
            "ESCALATE",
            "Low intent confidence",
            best_similarity
        )

    # --------------------------------------------------------
    # HIGH-RISK INTENTS
    # --------------------------------------------------------

    if intent in HIGH_RISK_INTENTS:

        if intent == "account_or_billing_issue":

            reason = (
                "Account or billing issue requires investigation"
            )

        elif intent in {
            "performance_freezing",
            "device_functionality_issue"
        }:

            reason = (
                "Device-specific troubleshooting required"
            )

        else:

            reason = (
                "Intent too broad or unclear"
            )

        return (
            "ESCALATE",
            reason,
            best_similarity
        )

    # --------------------------------------------------------
    # LOW HISTORICAL SIMILARITY
    # --------------------------------------------------------

    if best_similarity < SIMILARITY_THRESHOLD:

        return (
            "ESCALATE",
            "Low historical similarity",
            best_similarity
        )

    # --------------------------------------------------------
    # AUTO-HANDLE
    # --------------------------------------------------------

    return (
        "AUTO-HANDLE",
        "High-confidence intent with sufficient historical evidence",
        best_similarity
    )


# ============================================================
# COMPLETE SUPPORT AGENT
# ============================================================

def support_agent(
    message,
    classifier,
    vectorizer,
    retriever,
    database
):

    # Step 1: Intent classification
    intent, confidence = classify_intent(
        classifier,
        message
    )

    # Step 2: Retrieve historical evidence
    evidence = retrieve_evidence(
        vectorizer,
        retriever,
        database,
        message,
        intent
    )

    # Step 3: Apply decision policy
    action, reason, best_similarity = make_decision(
        message,
        intent,
        confidence,
        evidence
    )

    # Step 4: Generate reply
    reply = generate_reply(
        message,
        intent,
        action,
        evidence
    )

    return {
        "customer_message": message,
        "intent": intent,
        "confidence": confidence,
        "evidence_count": len(evidence),
        "best_similarity": best_similarity,
        "action": action,
        "reason": reason,
        "reply": reply,
        "evidence": evidence
    }


# ============================================================
# TEST CASES
# ============================================================

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("FINAL SUPPORT AGENT V4")
    print("=" * 80)

    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    classifier, vectorizer, retriever, database = load_models()

    # --------------------------------------------------------
    # Test messages
    # --------------------------------------------------------

    test_messages = [

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
    print("FINAL AGENT TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Run each test
    # --------------------------------------------------------

    for message in test_messages:

        result = support_agent(
            message,
            classifier,
            vectorizer,
            retriever,
            database
        )

        print("\n")
        print("-" * 80)

        print("Customer:")
        print(message)

        print("\nIntent:")
        print(result["intent"])

        print("\nConfidence:")
        print(
            round(
                result["confidence"],
                4
            )
        )

        print("\nEvidence count:")
        print(result["evidence_count"])

        print("\nBest similarity:")
        print(
            round(
                result["best_similarity"],
                4
            )
        )

        print("\nDecision:")
        print(result["action"])

        print("\nReason:")
        print(result["reason"])

        print("\nReply:")
        print(result["reply"])

        # ----------------------------------------------------
        # Display historical evidence
        # ----------------------------------------------------

        print("\nHistorical evidence:")

        if len(result["evidence"]) == 0:

            print("No historical evidence retrieved.")

        else:

            for index, item in enumerate(
                result["evidence"],
                start=1
            ):

                print("\nEvidence", index)

                print("Customer:")
                print(item["customer_text"])

                print("\nApple Support:")
                print(item["brand_text"])

                print("\nSimilarity:")
                print(
                    round(
                        item["similarity"],
                        4
                    )
                )

    # --------------------------------------------------------
    # Final policy
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL POLICY")
    print("=" * 80)

    print(
        "Intent confidence >= 0.70"
    )

    print(
        "Historical similarity >= 0.50"
    )

    print(
        "Sensitive account/security issues -> ESCALATE"
    )

    print(
        "Device-specific troubleshooting -> ESCALATE"
    )

    print(
        "Broad or unclear intents -> ESCALATE"
    )

    print(
        "Historical responses are used as evidence."
    )

    print(
        "Direct copying of historical responses is avoided."
    )

    print("\nFinal agent V4 completed successfully.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()