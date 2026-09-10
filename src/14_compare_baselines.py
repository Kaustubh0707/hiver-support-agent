# (First one)
# import os
# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support
# )


# # ============================================================
# # FILE PATHS
# # ============================================================

# INPUT_FILE = "data/processed/training_data.csv"

# MODEL_FILE = "models/intent_classifier.joblib"

# OUTPUT_FOLDER = "outputs"

# OUTPUT_FILE = os.path.join(
#     OUTPUT_FOLDER,
#     "baseline_comparison.csv"
# )


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 80)
#     print("HIVER AI SUPPORT AGENT")
#     print("BASELINE COMPARISON")
#     print("=" * 80)


#     # --------------------------------------------------------
#     # 1. Check files
#     # --------------------------------------------------------

#     if not os.path.exists(INPUT_FILE):

#         print("\nERROR: Training data not found.")

#         print(
#             "Expected:",
#             INPUT_FILE
#         )

#         return


#     if not os.path.exists(MODEL_FILE):

#         print("\nERROR: Classifier model not found.")

#         print(
#             "Expected:",
#             MODEL_FILE
#         )

#         return


#     # --------------------------------------------------------
#     # 2. Load data
#     # --------------------------------------------------------

#     print("\nLoading data...")

#     df = pd.read_csv(
#         INPUT_FILE,
#         low_memory=False
#     )


#     df["customer_text"] = (
#         df["customer_text"]
#         .fillna("")
#         .astype(str)
#         .str.strip()
#     )

#     df["intent"] = (
#         df["intent"]
#         .fillna("")
#         .astype(str)
#         .str.strip()
#     )


#     df = df[
#         (df["customer_text"] != "")
#         & (df["intent"] != "")
#     ].copy()


#     print(
#         "Usable examples:",
#         len(df)
#     )


#     # --------------------------------------------------------
#     # 3. Create same evaluation split
#     # --------------------------------------------------------

#     X = df["customer_text"]

#     y = df["intent"]


#     X_train, X_test, y_train, y_test = train_test_split(

#         X,

#         y,

#         test_size=0.20,

#         random_state=42,

#         stratify=y
#     )


#     print(
#         "\nTraining examples:",
#         len(X_train)
#     )

#     print(
#         "Evaluation examples:",
#         len(X_test)
#     )


#     # ========================================================
#     # BASELINE 1
#     # MAJORITY CLASS
#     # ========================================================

#     print("\n")
#     print("=" * 80)
#     print("BASELINE 1 - MAJORITY CLASS")
#     print("=" * 80)


#     majority_class = (
#         y_train
#         .value_counts()
#         .idxmax()
#     )


#     print(
#         "\nMajority intent:",
#         majority_class
#     )


#     majority_predictions = [
#         majority_class
#         for _ in range(len(y_test))
#     ]


#     majority_accuracy = accuracy_score(
#         y_test,
#         majority_predictions
#     )


#     majority_precision, majority_recall, majority_f1, _ = (
#         precision_recall_fscore_support(
#             y_test,
#             majority_predictions,
#             average="macro",
#             zero_division=0
#         )
#     )


#     print(
#         "\nAccuracy:",
#         round(majority_accuracy, 4)
#     )

#     print(
#         "Macro Precision:",
#         round(majority_precision, 4)
#     )

#     print(
#         "Macro Recall:",
#         round(majority_recall, 4)
#     )

#     print(
#         "Macro F1:",
#         round(majority_f1, 4)
#     )


#     # ========================================================
#     # BASELINE 2
#     # TF-IDF + LOGISTIC REGRESSION
#     # ========================================================

#     print("\n")
#     print("=" * 80)
#     print("BASELINE 2 - TF-IDF + LOGISTIC REGRESSION")
#     print("=" * 80)


#     model = joblib.load(
#         MODEL_FILE
#     )


#     print(
#         "\nGenerating predictions..."
#     )


#     ml_predictions = model.predict(
#         X_test
#     )


#     ml_accuracy = accuracy_score(
#         y_test,
#         ml_predictions
#     )


#     ml_precision, ml_recall, ml_f1, _ = (
#         precision_recall_fscore_support(
#             y_test,
#             ml_predictions,
#             average="macro",
#             zero_division=0
#         )
#     )


#     print(
#         "\nAccuracy:",
#         round(ml_accuracy, 4)
#     )

#     print(
#         "Macro Precision:",
#         round(ml_precision, 4)
#     )

#     print(
#         "Macro Recall:",
#         round(ml_recall, 4)
#     )

#     print(
#         "Macro F1:",
#         round(ml_f1, 4)
#     )


#     # ========================================================
#     # COMPARE RESULTS
#     # ========================================================

#     comparison_df = pd.DataFrame({

#         "system": [

#             "Majority Class Baseline",

#             "TF-IDF + Logistic Regression"
#         ],

#         "accuracy": [

#             majority_accuracy,

#             ml_accuracy
#         ],

#         "macro_precision": [

#             majority_precision,

#             ml_precision
#         ],

#         "macro_recall": [

#             majority_recall,

#             ml_recall
#         ],

#         "macro_f1": [

#             majority_f1,

#             ml_f1
#         ]
#     })


#     # --------------------------------------------------------
#     # Calculate improvement
#     # --------------------------------------------------------

#     accuracy_improvement = (
#         ml_accuracy -
#         majority_accuracy
#     )


#     f1_improvement = (
#         ml_f1 -
#         majority_f1
#     )


#     print("\n")
#     print("=" * 80)
#     print("BASELINE COMPARISON")
#     print("=" * 80)


#     print(
#         comparison_df.to_string(
#             index=False
#         )
#     )


#     print("\n")
#     print(
#         "Accuracy improvement over majority baseline:",
#         round(
#             accuracy_improvement,
#             4
#         )
#     )


#     print(
#         "Macro F1 improvement over majority baseline:",
#         round(
#             f1_improvement,
#             4
#         )
#     )


#     # --------------------------------------------------------
#     # Save results
#     # --------------------------------------------------------

#     os.makedirs(
#         OUTPUT_FOLDER,
#         exist_ok=True
#     )


#     comparison_df.to_csv(
#         OUTPUT_FILE,
#         index=False
#     )


#     print("\n")
#     print("=" * 80)
#     print("BASELINE RESULTS SAVED")
#     print("=" * 80)


#     print(
#         "\nFile:"
#     )

#     print(
#         OUTPUT_FILE
#     )


#     print("\n")
#     print(
#         "IMPORTANT:"
#     )

#     print(
#         """
# These results use automatically generated
# pseudo-labels. They should be described as
# development/baseline results rather than
# human-validated customer-support performance.
# """
#     )


# if __name__ == "__main__":

#     main()

# (New One)
import os
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


INPUT_FILE = "data/processed/training_data.csv"
OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "baseline_comparison.csv"
)


def keyword_baseline(text):
    text = str(text).lower()

    # Battery
    if any(word in text for word in [
        "battery",
        "charging",
        "charger",
        "overheating",
        "overheat"
    ]):
        return "battery_issue"

    # Software update
    if any(word in text for word in [
        "ios update",
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17",
        "ios 18",
        "software update",
        "update",
        "updating",
        "updated",
        "upgrade",
        "downgrade"
    ]):
        return "software_update_issue"

    # Performance / freezing
    if any(word in text for word in [
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
    ]):
        return "performance_freezing"

    # Apps / iTunes
    if any(word in text for word in [
        "itunes",
        "app store",
        "appstore",
        "application",
        "apps",
        "download app",
        "install app",
        "apple music",
        "music app"
    ]):
        return "app_or_itunes_issue"

    # Connectivity
    if any(word in text for word in [
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
    ]):
        return "connectivity_issue"

    # Account / billing
    if any(word in text for word in [
        "account",
        "apple id",
        "appleid",
        "password",
        "login",
        "log in",
        "sign in",
        "charged",
        "payment",
        "billing",
        "bill",
        "refund",
        "purchase",
        "subscription",
        "money"
    ]):
        return "account_or_billing_issue"

    # Device functionality
    if any(word in text for word in [
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
    ]):
        return "device_functionality_issue"

    # Information / how-to
    if any(word in text for word in [
        "how do i",
        "how to",
        "where can i",
        "can i",
        "is there a way",
        "what is",
        "what does",
        "which",
        "how can i"
    ]):
        return "information_or_how_to"

    # General support
    if any(word in text for word in [
        "help",
        "please help",
        "support",
        "issue",
        "problem",
        "not working",
        "doesn't work",
        "does not work"
    ]):
        return "general_support"

    return "other"


def calculate_metrics(y_true, predictions):

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "macro_precision": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )
    }


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("BASELINE COMPARISON")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Training data not found.")

        print("\nExpected:")
        print(INPUT_FILE)

        print("\nPlease run:")
        print("python src/07_create_training_data.py")

        return

    print("\nLoading development dataset...")

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

    X = df["customer_text"]
    y = df["intent"]

    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training examples:", len(X_train))
    print("Testing examples:", len(X_test))

    results = []

    # =========================================================
    # 1. TRIVIAL BASELINE
    # =========================================================

    print("\n")
    print("-" * 80)
    print("1. MAJORITY CLASS BASELINE")
    print("-" * 80)

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(
        X_train.to_frame(),
        y_train
    )

    dummy_predictions = dummy.predict(
        X_test.to_frame()
    )

    dummy_metrics = calculate_metrics(
        y_test,
        dummy_predictions
    )

    print(
        "Accuracy:",
        round(dummy_metrics["accuracy"], 4)
    )

    print(
        "Macro Precision:",
        round(dummy_metrics["macro_precision"], 4)
    )

    print(
        "Macro Recall:",
        round(dummy_metrics["macro_recall"], 4)
    )

    print(
        "Macro F1:",
        round(dummy_metrics["macro_f1"], 4)
    )

    results.append({
        "system": "Majority Class Baseline",
        **dummy_metrics
    })

    # =========================================================
    # 2. SIMPLE KEYWORD BASELINE
    # =========================================================

    print("\n")
    print("-" * 80)
    print("2. KEYWORD RULE BASELINE")
    print("-" * 80)

    keyword_predictions = X_test.apply(
        keyword_baseline
    )

    keyword_metrics = calculate_metrics(
        y_test,
        keyword_predictions
    )

    print(
        "Accuracy:",
        round(keyword_metrics["accuracy"], 4)
    )

    print(
        "Macro Precision:",
        round(keyword_metrics["macro_precision"], 4)
    )

    print(
        "Macro Recall:",
        round(keyword_metrics["macro_recall"], 4)
    )

    print(
        "Macro F1:",
        round(keyword_metrics["macro_f1"], 4)
    )

    results.append({
        "system": "Keyword Rule Baseline",
        **keyword_metrics
    })

    # =========================================================
    # 3. MAIN MACHINE LEARNING MODEL
    # =========================================================

    print("\n")
    print("-" * 80)
    print("3. TF-IDF + LOGISTIC REGRESSION")
    print("-" * 80)

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=20000
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test
    )

    model_metrics = calculate_metrics(
        y_test,
        predictions
    )

    print(
        "Accuracy:",
        round(model_metrics["accuracy"], 4)
    )

    print(
        "Macro Precision:",
        round(model_metrics["macro_precision"], 4)
    )

    print(
        "Macro Recall:",
        round(model_metrics["macro_recall"], 4)
    )

    print(
        "Macro F1:",
        round(model_metrics["macro_f1"], 4)
    )

    results.append({
        "system": "TF-IDF + Logistic Regression",
        **model_metrics
    })

    # =========================================================
    # FINAL COMPARISON
    # =========================================================

    results_df = pd.DataFrame(results)

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
    print("FINAL BASELINE COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(index=False)
    )

    print("\n")
    print("Saved results:")
    print(OUTPUT_FILE)

    # =========================================================
    # IMPORTANT INTERPRETATION
    # =========================================================

    print("\n")
    print("=" * 80)
    print("IMPORTANT INTERPRETATION")
    print("=" * 80)

    print(
        "\nThe keyword baseline may perform very strongly because "
        "the development labels were generated using keyword-based "
        "rules."
    )

    print(
        "\nTherefore, these development metrics should not be "
        "interpreted as human-validated intent accuracy."
    )

    print(
        "\nA human-labelled evaluation set is required to measure "
        "true intent classification performance."
    )


if __name__ == "__main__":
    main()