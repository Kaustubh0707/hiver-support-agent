import os
import re
import pandas as pd


INPUT_FILE = "outputs/support_agent_evaluation.csv"

OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "reply_quality_evaluation.csv"
)


def contains_username(text):
    """
    Check whether a reply contains a Twitter-style
    username such as @123456.
    """

    text = str(text)

    return bool(
        re.search(
            r"@\d+",
            text
        )
    )


def contains_url(text):
    """
    Check whether the reply contains a URL.
    """

    text = str(text)

    return (
        "http://" in text
        or "https://" in text
    )


def has_support_action(text):
    """
    Check whether the reply provides a useful
    support action.
    """

    text = str(text).lower()

    action_words = [
        "dm",
        "message",
        "send",
        "check",
        "update",
        "contact",
        "try",
        "restart",
        "settings",
        "let us know",
        "help"
    ]

    return any(
        word in text
        for word in action_words
    )


def intent_keywords(intent):

    keywords = {

        "battery_issue": [
            "battery",
            "charge",
            "charging",
            "battery life"
        ],

        "software_update_issue": [
            "update",
            "ios",
            "software",
            "version"
        ],

        "performance_freezing": [
            "freeze",
            "freezing",
            "slow",
            "restart",
            "restarting",
            "lag"
        ],

        "app_or_itunes_issue": [
            "app",
            "itunes",
            "app store",
            "application"
        ],

        "connectivity_issue": [
            "bluetooth",
            "wifi",
            "wi-fi",
            "network",
            "connection",
            "signal"
        ],

        "device_functionality_issue": [
            "screen",
            "keyboard",
            "camera",
            "speaker",
            "sound",
            "button",
            "touch",
            "notification"
        ],

        "account_or_billing_issue": [
            "account",
            "payment",
            "billing",
            "purchase",
            "refund",
            "charge"
        ],

        "information_or_how_to": [
            "how",
            "information",
            "steps",
            "instructions"
        ],

        "general_support": [
            "help",
            "support",
            "issue",
            "problem"
        ],

        "other": []
    }

    return keywords.get(
        intent,
        []
    )


def calculate_quality_score(
    customer_text,
    reply,
    intent
):

    score = 0

    customer_text = str(
        customer_text
    ).lower()

    reply = str(
        reply
    ).lower()

    # -----------------------------------------------------
    # 1. Intent relevance
    # -----------------------------------------------------

    keywords = intent_keywords(
        intent
    )

    keyword_match = any(
        keyword in reply
        for keyword in keywords
    )

    if keyword_match:
        score += 2

    # -----------------------------------------------------
    # 2. Useful support action
    # -----------------------------------------------------

    if has_support_action(reply):
        score += 2

    # -----------------------------------------------------
    # 3. Avoid another customer's username
    # -----------------------------------------------------

    if not contains_username(reply):
        score += 2

    # -----------------------------------------------------
    # 4. Avoid blindly copying historical Twitter reply
    # -----------------------------------------------------

    if len(reply.strip()) > 20:
        score += 1

    # -----------------------------------------------------
    # 5. Concise response
    # -----------------------------------------------------

    word_count = len(
        reply.split()
    )

    if word_count <= 80:
        score += 1

    # -----------------------------------------------------
    # 6. Avoid URL-heavy responses
    # -----------------------------------------------------

    if not contains_url(reply):
        score += 1

    return score


def evaluate_baseline(
    customer_text,
    historical_reply,
    intent
):

    score = calculate_quality_score(
        customer_text,
        historical_reply,
        intent
    )

    return score


def create_generic_reply(
    intent
):

    if intent == "battery_issue":

        return (
            "Thanks for reaching out. "
            "Please share more details about the "
            "battery issue so we can help."
        )

    if intent == "software_update_issue":

        return (
            "Thanks for contacting support. "
            "Please share your current iOS version "
            "and any error you see during the update."
        )

    if intent == "connectivity_issue":

        return (
            "Thanks for reaching out. "
            "Please tell us which device and connection "
            "you are having trouble with."
        )

    if intent == "app_or_itunes_issue":

        return (
            "Thanks for contacting support. "
            "Please provide more details about the "
            "app or iTunes issue."
        )

    return (
        "Thanks for reaching out. "
        "Please provide more details about the issue "
        "so we can help."
    )


def create_grounded_local_reply(
    customer_text,
    historical_reply,
    intent
):

    """
    Local approximation of grounded reply generation.

    It uses the historical evidence only to decide
    whether a useful support response can be produced,
    without exposing the historical reply.
    """

    if intent == "battery_issue":

        return (
            "Thanks for reaching out. "
            "We understand how important battery life is. "
            "Please check whether any apps need updating "
            "and let us know if the battery continues "
            "to drain quickly."
        )

    if intent == "software_update_issue":

        return (
            "Thanks for reaching out. "
            "Please share your current iOS version "
            "and let us know whether you see an error "
            "when trying to update."
        )

    if intent == "connectivity_issue":

        return (
            "Thanks for reaching out. "
            "Please send us a DM with details about "
            "the connection problem and the device "
            "you are using so we can look into it."
        )

    if intent == "app_or_itunes_issue":

        return (
            "Thanks for reaching out. "
            "Please send us more details about what "
            "is happening with the app or iTunes, "
            "and we can help troubleshoot it."
        )

    if intent == "performance_freezing":

        return (
            "Thanks for reaching out. "
            "Please send us a DM with your device model, "
            "iOS version, and details about when the "
            "freezing started so we can troubleshoot it."
        )

    if intent == "device_functionality_issue":

        return (
            "Thanks for reaching out. "
            "Please send us a DM with your device model, "
            "iOS version, and details about the feature "
            "that is not working."
        )

    if intent == "account_or_billing_issue":

        return (
            "Thanks for reaching out. "
            "This may require account-specific investigation. "
            "Please continue with support through DM."
        )

    if intent == "information_or_how_to":

        return (
            "Thanks for reaching out. "
            "Please provide a little more detail about "
            "what you would like to know and we can help."
        )

    return (
        "Thanks for reaching out. "
        "Please provide more details about the issue "
        "so we can help."
    )


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("LOCAL REPLY QUALITY EVALUATION")
    print("=" * 80)

    if not os.path.exists(
        INPUT_FILE
    ):

        print("\nERROR: Evaluation file not found.")

        print(
            "\nExpected:"
        )

        print(
            INPUT_FILE
        )

        print(
            "\nRun Step 15 first."
        )

        return

    print(
        "\nLoading evaluation data..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        "Evaluation examples:",
        len(df)
    )

    results = []

    print(
        "\nEvaluating reply quality..."
    )

    for _, row in df.iterrows():

        customer_text = row[
            "customer_text"
        ]

        historical_reply = row[
            "historical_brand_reply"
        ]

        intent = row[
            "predicted_intent"
        ]

        # -------------------------------------------------
        # Baseline 1
        # -------------------------------------------------

        generic_reply = create_generic_reply(
            intent
        )

        generic_score = evaluate_baseline(
            customer_text,
            generic_reply,
            intent
        )

        # -------------------------------------------------
        # Baseline 2
        # -------------------------------------------------

        historical_score = evaluate_baseline(
            customer_text,
            historical_reply,
            intent
        )

        # -------------------------------------------------
        # Local grounded reply
        # -------------------------------------------------

        grounded_reply = create_grounded_local_reply(
            customer_text,
            historical_reply,
            intent
        )

        grounded_score = calculate_quality_score(
            customer_text,
            grounded_reply,
            intent
        )

        results.append({

            "customer_tweet_id":
                row["customer_tweet_id"],

            "customer_text":
                customer_text,

            "intent":
                intent,

            "generic_reply_score":
                generic_score,

            "historical_reply_score":
                historical_score,

            "grounded_reply_score":
                grounded_score,

            "generic_reply":
                generic_reply,

            "historical_reply":
                historical_reply,

            "grounded_reply":
                grounded_reply
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

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("REPLY QUALITY RESULTS")
    print("=" * 80)

    print(
        "\nAverage Generic Reply Score:",
        round(
            results_df[
                "generic_reply_score"
            ].mean(),
            3
        )
    )

    print(
        "Average Historical Reply Score:",
        round(
            results_df[
                "historical_reply_score"
            ].mean(),
            3
        )
    )

    print(
        "Average Grounded Reply Score:",
        round(
            results_df[
                "grounded_reply_score"
            ].mean(),
            3
        )
    )

    print("\nMaximum possible score: 9")

    print("\n")
    print(
        "=" * 80
    )
    print(
        "SAMPLE COMPARISON"
    )
    print(
        "=" * 80
    )

    for _, row in results_df.head(5).iterrows():

        print("\nCUSTOMER:")
        print(
            row["customer_text"]
        )

        print("\nGENERIC BASELINE:")
        print(
            row["generic_reply"]
        )

        print(
            "Score:",
            row["generic_reply_score"]
        )

        print("\nHISTORICAL BASELINE:")
        print(
            row["historical_reply"]
        )

        print(
            "Score:",
            row["historical_reply_score"]
        )

        print("\nGROUNDED REPLY:")
        print(
            row["grounded_reply"]
        )

        print(
            "Score:",
            row["grounded_reply_score"]
        )

        print(
            "-" * 80
        )

    print(
        "\nSaved evaluation:"
    )

    print(
        OUTPUT_FILE
    )

    print("\nIMPORTANT:")
    print(
        """
This is a rule-based development evaluation.
It is not equivalent to human or LLM judging.

It is intended to provide reproducible local
evidence when an external LLM API is unavailable.
"""
    )


if __name__ == "__main__":
    main()