import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "data/processed/training_data.csv"


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("STRONGER CLASSIFIER EXPERIMENT")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load training data
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Training data not found.")
        print("Expected file:")
        print(INPUT_FILE)
        return

    print("\nLoading training data...")

    df = pd.read_csv(INPUT_FILE)

    print("Total examples:", len(df))

    # ---------------------------------------------------------
    # 2. Check required columns
    # ---------------------------------------------------------

    required_columns = [
        "customer_text",
        "intent"
    ]

    for column in required_columns:
        if column not in df.columns:
            print("\nERROR: Missing column:", column)
            return

    # ---------------------------------------------------------
    # 3. Clean data
    # ---------------------------------------------------------

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

    print("Examples after cleaning:", len(df))

    # ---------------------------------------------------------
    # 4. Features and labels
    # ---------------------------------------------------------

    X = df["customer_text"]
    y = df["intent"]

    # ---------------------------------------------------------
    # 5. Train-test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining examples:", len(X_train))
    print("Testing examples:", len(X_test))

    # ---------------------------------------------------------
    # 6. Stronger TF-IDF + Linear SVM
    # ---------------------------------------------------------

    print("\nTraining TF-IDF + Linear SVM...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=20000
            )
        ),
        (
            "classifier",
            LinearSVC(
                C=1.5,
                class_weight="balanced"
            )
        )
    ])

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 7. Predictions
    # ---------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    # ---------------------------------------------------------
    # 8. Evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print(f"\nAccuracy percentage:")
    print(f"{accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print("\nPrevious TF-IDF + Logistic Regression:")
    print("Accuracy: 79.10%")

    print("\nNew TF-IDF + Linear SVM:")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    improvement = (accuracy - 0.791) * 100

    print("\nDifference:")
    print(f"{improvement:+.2f} percentage points")

    if accuracy > 0.791:
        print("\nRESULT: New classifier is better than the current model.")
    elif accuracy < 0.791:
        print("\nRESULT: Current Logistic Regression model is better.")
    else:
        print("\nRESULT: Both models have the same accuracy.")

    print("\nIMPORTANT:")
    print("This evaluation uses automatically generated pseudo-labels.")
    print("It is a development experiment, not a human-validated benchmark.")


if __name__ == "__main__":
    main()