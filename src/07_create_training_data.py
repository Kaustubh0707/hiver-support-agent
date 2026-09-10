import os
import re
import pandas as pd


INPUT_FILE = "data/processed/selected_brand_data.csv"
OUTPUT_FOLDER = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "training_data.csv")

SAMPLES_PER_INTENT = 1000


def classify_intent(text):
    """
    Automatically assign an intent using keyword-based rules.
    """

    text = str(text).lower()

    # 1. Battery issues
    battery_words = [
        "battery",
        "battery life",
        "battery drain",
        "draining battery",
        "charge",
        "charging",
        "charger",
        "overheating",
        "overheat"
    ]

    if any(word in text for word in battery_words):
        return "battery_issue"


    # 2. Software update issues
    update_words = [
        "ios update",
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17",
        "ios 18",
        "update",
        "updating",
        "updated",
        "upgrade",
        "downgrade",
        "install ios",
        "software update"
    ]

    if any(word in text for word in update_words):
        return "software_update_issue"


    # 3. Performance / freezing / restarting
    performance_words = [
        "freeze",
        "freezing",
        "frozen",
        "crash",
        "crashing",
        "restart",
        "restarting",
        "reboot",
        "slow",
        "lag",
        "lagging",
        "stuck",
        "unresponsive"
    ]

    if any(word in text for word in performance_words):
        return "performance_freezing"


    # 4. App / iTunes / App Store
    app_words = [
        "itunes",
        "app store",
        "appstore",
        "application",
        "app ",
        "apps",
        "download app",
        "install app",
        "apple music",
        "music app"
    ]

    if any(word in text for word in app_words):
        return "app_or_itunes_issue"


    # 5. Connectivity
    connectivity_words = [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "mobile data",
        "network",
        "no service",
        "signal",
        "connection",
        "connect",
        "calling",
        "call drop"
    ]

    if any(word in text for word in connectivity_words):
        return "connectivity_issue"


    # 6. Account / billing / purchases
    account_words = [
        "account",
        "apple id",
        "appleid",
        "password",
        "login",
        "log in",
        "sign in",
        "charge",
        "charged",
        "payment",
        "billing",
        "bill",
        "refund",
        "purchase",
        "subscription",
        "money"
    ]

    if any(word in text for word in account_words):
        return "account_or_billing_issue"


    # 7. Device functionality
    functionality_words = [
        "keyboard",
        "screen",
        "display",
        "camera",
        "speaker",
        "sound",
        "volume",
        "microphone",
        "button",
        "touch",
        "touchscreen",
        "face id",
        "touch id",
        "home button",
        "notification"
    ]

    if any(word in text for word in functionality_words):
        return "device_functionality_issue"


    # 8. Information / how-to
    information_words = [
        "how do i",
        "how to",
        "where can i",
        "can i",
        "is there a way",
        "what is",
        "what does",
        "which",
        "help me understand",
        "how can i"
    ]

    if any(word in text for word in information_words):
        return "information_or_how_to"


    # 9. General support
    general_words = [
        "help",
        "please help",
        "support",
        "issue",
        "problem",
        "not working",
        "doesn't work",
        "does not work"
    ]

    if any(word in text for word in general_words):
        return "general_support"


    # 10. Other
    return "other"


def decide_auto_handle(text, intent):
    """
    Automatically decide whether the issue can be handled
    without human investigation.
    """

    text = str(text).lower()

    # These cases generally require investigation.
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

    if any(word in text for word in escalation_words):
        return "NO"

    # General vague support requests should go to a human.
    if intent == "general_support":
        return "NO"

    # Other/unclear messages should go to a human.
    if intent == "other":
        return "NO"

    # Device-specific failures can require troubleshooting.
    if intent == "device_functionality_issue":
        return "NO"

    # Performance problems can require troubleshooting.
    if intent == "performance_freezing":
        return "NO"

    return "YES"


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("AUTOMATIC TRAINING DATA CREATION")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Selected brand dataset not found.")
        print("Expected file:")
        print(INPUT_FILE)
        return

    print("\nLoading AppleSupport conversations...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("Total conversations:", len(df))

    # Clean text
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

    df = df[df["customer_text"] != ""].copy()

    # Remove duplicate customer messages
    df = df.drop_duplicates(subset=["customer_text"])

    print("Unique customer messages:", len(df))

    print("\nAutomatically classifying intents...")

    df["intent"] = df["customer_text"].apply(classify_intent)

    print("\nIntent distribution BEFORE sampling:")
    print("-" * 60)
    print(df["intent"].value_counts())

    print("\nCreating auto-handle decisions...")

    df["auto_handle"] = df.apply(
        lambda row: decide_auto_handle(
            row["customer_text"],
            row["intent"]
        ),
        axis=1
    )

    # Create a balanced training dataset.
    print("\nCreating balanced training dataset...")

    training_parts = []

    for intent in df["intent"].unique():

        intent_df = df[df["intent"] == intent]

        sample_size = min(
            SAMPLES_PER_INTENT,
            len(intent_df)
        )

        if sample_size > 0:
            sampled = intent_df.sample(
                n=sample_size,
                random_state=42
            )

            training_parts.append(sampled)

    training_df = pd.concat(
        training_parts,
        ignore_index=True
    )

    # Shuffle final dataset
    training_df = training_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Add explanation for automatically generated labels
    training_df["labeling_notes"] = (
        "Automatically labeled using keyword-based intent rules."
    )

    # Keep useful columns
    training_df = training_df[
        [
            "customer_tweet_id",
            "customer_text",
            "brand_text",
            "intent",
            "auto_handle",
            "labeling_notes"
        ]
    ]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    training_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("TRAINING DATA CREATED SUCCESSFULLY")
    print("=" * 80)

    print("\nTraining examples:", len(training_df))

    print("\nFinal intent distribution:")
    print("-" * 60)
    print(training_df["intent"].value_counts())

    print("\nAuto-handle distribution:")
    print("-" * 60)
    print(training_df["auto_handle"].value_counts())

    print("\nSaved file:")
    print(OUTPUT_FILE)

    print("\nNext step:")
    print("Build the TF-IDF + Logistic Regression classifier.")


if __name__ == "__main__":
    main()