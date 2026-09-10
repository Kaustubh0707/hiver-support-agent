import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/processed/training_data.csv"

MODEL_FILE = "models/intent_classifier.joblib"

OUTPUT_FOLDER = "outputs"

METRICS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "classifier_metrics.csv"
)

CONFUSION_FILE = os.path.join(
    OUTPUT_FOLDER,
    "classifier_confusion_matrix.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("INTENT CLASSIFIER EVALUATION")
    print("=" * 80)


    # --------------------------------------------------------
    # 1. Check required files
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Training data not found.")

        print(
            "\nExpected:"
        )

        print(INPUT_FILE)

        return


    if not os.path.exists(MODEL_FILE):

        print("\nERROR: Classifier model not found.")

        print(
            "\nExpected:"
        )

        print(MODEL_FILE)

        print(
            "\nPlease run:"
        )

        print(
            "python src/08_train_intent_classifier.py"
        )

        return


    # --------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------

    print("\nLoading evaluation data...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        "Total examples:",
        len(df)
    )


    # --------------------------------------------------------
    # 3. Clean data
    # --------------------------------------------------------

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
        (df["customer_text"] != "")
        & (df["intent"] != "")
    ].copy()


    print(
        "Usable examples:",
        len(df)
    )


    # --------------------------------------------------------
    # 4. Create same validation split
    # --------------------------------------------------------

    print(
        "\nCreating evaluation split..."
    )

    X = df["customer_text"]

    y = df["intent"]


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    print(
        "Training examples:",
        len(X_train)
    )

    print(
        "Evaluation examples:",
        len(X_test)
    )


    # --------------------------------------------------------
    # 5. Load trained classifier
    # --------------------------------------------------------

    print(
        "\nLoading trained classifier..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Classifier loaded successfully."
    )


    # --------------------------------------------------------
    # 6. Generate predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # 7. Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # 8. Print headline metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CLASSIFIER EVALUATION RESULTS")
    print("=" * 80)


    print(
        "\nAccuracy:",
        round(accuracy, 4)
    )


    print(
        "Macro Precision:",
        round(precision_macro, 4)
    )


    print(
        "Macro Recall:",
        round(recall_macro, 4)
    )


    print(
        "Macro F1:",
        round(f1_macro, 4)
    )


    print(
        "Weighted Precision:",
        round(precision_weighted, 4)
    )


    print(
        "Weighted Recall:",
        round(recall_weighted, 4)
    )


    print(
        "Weighted F1:",
        round(f1_weighted, 4)
    )


    # --------------------------------------------------------
    # 9. Detailed classification report
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)


    report_text = classification_report(

        y_test,

        predictions,

        zero_division=0
    )


    print(
        report_text
    )


    # --------------------------------------------------------
    # 10. Confusion matrix
    # --------------------------------------------------------

    labels = sorted(
        y_test.unique()
    )


    matrix = confusion_matrix(

        y_test,

        predictions,

        labels=labels
    )


    confusion_df = pd.DataFrame(

        matrix,

        index=labels,

        columns=labels
    )


    # --------------------------------------------------------
    # 11. Find most confused intent pairs
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TOP CONFUSIONS")
    print("=" * 80)


    confusion_pairs = []


    for actual_index in range(
        len(labels)
    ):

        for predicted_index in range(
            len(labels)
        ):

            if actual_index == predicted_index:
                continue


            count = matrix[
                actual_index,
                predicted_index
            ]


            if count > 0:

                confusion_pairs.append(

                    (
                        count,

                        labels[actual_index],

                        labels[predicted_index]
                    )
                )


    confusion_pairs.sort(
        reverse=True
    )


    for count, actual, predicted in confusion_pairs[:10]:

        print(
            f"{actual} -> {predicted}: {count}"
        )


    # --------------------------------------------------------
    # 12. Save metrics
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )


    metrics_df = pd.DataFrame({

        "metric": [

            "accuracy",

            "macro_precision",

            "macro_recall",

            "macro_f1",

            "weighted_precision",

            "weighted_recall",

            "weighted_f1"
        ],

        "value": [

            accuracy,

            precision_macro,

            recall_macro,

            f1_macro,

            precision_weighted,

            recall_weighted,

            f1_weighted
        ]
    })


    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )


    confusion_df.to_csv(
        CONFUSION_FILE
    )


    # --------------------------------------------------------
    # 13. Save prediction-level evaluation data
    # --------------------------------------------------------

    evaluation_df = pd.DataFrame({

        "customer_text": X_test.values,

        "actual_intent": y_test.values,

        "predicted_intent": predictions

    })


    prediction_file = os.path.join(

        OUTPUT_FOLDER,

        "classifier_predictions.csv"
    )


    evaluation_df.to_csv(

        prediction_file,

        index=False
    )


    # --------------------------------------------------------
    # 14. Final output
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("EVALUATION FILES CREATED")
    print("=" * 80)


    print(
        "\nMetrics:"
    )

    print(
        METRICS_FILE
    )


    print(
        "\nConfusion matrix:"
    )

    print(
        CONFUSION_FILE
    )


    print(
        "\nPrediction-level results:"
    )

    print(
        prediction_file
    )


    print("\n")
    print("=" * 80)
    print("IMPORTANT INTERPRETATION")
    print("=" * 80)


    print(
        """
These labels were automatically generated using
keyword-based rules.

Therefore, these metrics measure agreement with
the pseudo-labeling process, not true human-labeled
customer-support intent accuracy.

Do NOT present this number as the final real-world
accuracy of the support agent.
"""
    )


if __name__ == "__main__":

    main()