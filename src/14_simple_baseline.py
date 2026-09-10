import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

INPUT_FILE = "data/processed/training_data.csv"
OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "simple_baseline_results.csv")


def keyword_baseline(text):
    text = str(text).lower()

    if any(word in text for word in [
        "battery", "charging", "charger", "overheating", "overheat"
    ]):
        return "battery_issue"

    if any(word in text for word in [
        "ios update", "ios 11", "ios 12", "ios 13",
        "ios 14", "ios 15", "ios 16", "ios 17",
        "ios 18", "software update", "update",
        "updating", "updated", "upgrade", "downgrade"
    ]):
        return "software_update_issue"

    if any(word in text for word in [
        "freeze", "freezing", "frozen", "crash",
        "crashing", "restart", "restarting",
        "reboot", "slow", "lag", "lagging",
        "stuck", "unresponsive"
    ]):
        return "performance_freezing"

    if any(word in text for word in [
        "itunes", "app store", "appstore",
        "application", "apps", "download app",
        "install app", "apple music", "music app"
    ]):
        return "app_or_itunes_issue"

    if any(word in text for word in [
        "wifi", "wi-fi", "bluetooth", "cellular",
        "mobile data", "network", "no service",
        "signal", "connection", "connect",
        "calling", "call drop"
    ]):
        return "connectivity_issue"

    if any(word in text for word in [
        "account", "apple id", "appleid", "password",
        "login", "log in", "sign in", "charged",
        "payment", "billing", "bill", "refund",
        "purchase", "subscription", "money"
    ]):
        return "account_or_billing_issue"

    if any(word in text for word in [
        "keyboard", "screen", "display", "camera",
        "speaker", "sound", "volume", "microphone",
        "button", "touch", "touchscreen", "face id",
        "touch id", "home button", "notification"
    ]):
        return "device_functionality_issue"

    if any(word in text for word in [
        "how do i", "how to", "where can i",
        "can i", "is there a way", "what is",
        "what does", "which", "how can i"
    ]):
        return "information_or_how_to"

    if any(word in text for word in [
        "help", "please help", "support",
        "issue", "problem", "not working",
        "doesn't work", "does not work"
    ]):
        return "general_support"

    return "other"


def main():
    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("SIMPLE KEYWORD BASELINE")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Training data not found.")
        print("Expected:")
        print(INPUT_FILE)
        return

    print("\nLoading development dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["customer_text"] != "") &
        (df["intent"] != "")
    ].copy()

    print("Examples:", len(df))

    print("\nRunning keyword baseline...")
    predictions = df["customer_text"].apply(keyword_baseline)

    accuracy = accuracy_score(df["intent"], predictions)

    macro_precision = precision_score(
        df["intent"],
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        df["intent"],
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        df["intent"],
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 80)
    print("SIMPLE BASELINE RESULTS")
    print("=" * 80)

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro Precision:", round(macro_precision, 4))
    print("Macro Recall:", round(macro_recall, 4))
    print("Macro F1:", round(macro_f1, 4))

    results = pd.DataFrame([
        {
            "system": "Keyword Rule Baseline",
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1
        }
    ])

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved results:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()