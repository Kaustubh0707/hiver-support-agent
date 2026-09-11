import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "data/processed/training_data.csv"


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("WORD + CHARACTER TF-IDF + LINEAR SVM EXPERIMENT")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Training data not found.")
        print("Expected file:")
        print(INPUT_FILE)
        return

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\nLoading training data...")

    df = pd.read_csv(INPUT_FILE)

    print("Total examples:", len(df))

    required_columns = ["customer_text", "intent"]

    for column in required_columns:
        if column not in df.columns:
            print("\nERROR: Missing column:", column)
            return

    # ---------------------------------------------------------
    # 2. Clean data
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
    # 3. Train/test split
    # ---------------------------------------------------------

    X = df["customer_text"]
    y = df["intent"]

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
    # 4. Word + Character TF-IDF
    # ---------------------------------------------------------

    print("\nCreating Word + Character TF-IDF features...")

    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=30000
            )
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )
        )
    ])

    # ---------------------------------------------------------
    # 5. Linear SVM
    # ---------------------------------------------------------

    print("\nTraining Word + Character TF-IDF + Linear SVM...")

    model = Pipeline([
        (
            "features",
            features
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
    # 6. Predictions
    # ---------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    # ---------------------------------------------------------
    # 7. Evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nAccuracy percentage:")
    print(f"{accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    # ---------------------------------------------------------
    # 8. Compare with previous models
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print("\nMajority baseline:")
    print("Accuracy: 10.00%")

    print("\nKeyword baseline:")
    print("Accuracy: 95.60%")

    print("\nPrevious TF-IDF + Logistic Regression:")
    print("Accuracy: 79.10%")

    print("\nPrevious Word TF-IDF + Linear SVM:")
    print("Accuracy: 85.50%")

    print("\nNew Word + Character TF-IDF + Linear SVM:")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    improvement = (accuracy - 0.855) * 100

    print("\nImprovement over previous SVM:")
    print(f"{improvement:+.2f} percentage points")

    # ---------------------------------------------------------
    # 9. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULT")
    print("=" * 80)

    if accuracy > 0.855:
        print("\nRESULT: NEW MODEL IS BETTER THAN THE 85.50% SVM.")

    elif accuracy < 0.855:
        print("\nRESULT: PREVIOUS 85.50% SVM IS BETTER.")

    else:
        print("\nRESULT: BOTH MODELS HAVE THE SAME ACCURACY.")

    print("\nIMPORTANT:")
    print("This evaluation uses automatically generated pseudo-labels.")
    print("It is a development experiment, not a human-validated benchmark.")


if __name__ == "__main__":
    main()