import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


INPUT_FILE = "data/processed/training_data.csv"
OUTPUT_FILE = "outputs/suspicious_training_examples.csv"


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("SUSPICIOUS TRAINING EXAMPLE ANALYSIS")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Training data not found.")
        print(INPUT_FILE)
        return

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\nLoading training data...")

    df = pd.read_csv(INPUT_FILE)

    print("Total examples:", len(df))

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

    # ---------------------------------------------------------
    # 2. Split data
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
    # 3. Create same strong model
    # ---------------------------------------------------------

    print("\nBuilding Word + Character TF-IDF model...")

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

    model = Pipeline([
        (
            "features",
            features
        ),
        (
            "classifier",
            LinearSVC(
                C=3.0,
                class_weight="balanced"
            )
        )
    ])

    print("\nTraining model...")

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4. Predict training examples
    # ---------------------------------------------------------

    print("\nPredicting training examples...")

    train_predictions = model.predict(X_train)

    # decision_function gives confidence-like scores
    decision_scores = model.decision_function(X_train)

    # ---------------------------------------------------------
    # 5. Find suspicious examples
    # ---------------------------------------------------------

    suspicious_rows = []

    classes = model.named_steps["classifier"].classes_

    for i, (text, actual, predicted) in enumerate(
        zip(X_train, y_train, train_predictions)
    ):

        scores = decision_scores[i]

        sorted_scores = sorted(scores, reverse=True)

        top_score = sorted_scores[0]

        second_score = sorted_scores[1]

        margin = top_score - second_score

        if actual != predicted:

            suspicious_rows.append({
                "customer_text": text,
                "original_intent": actual,
                "model_prediction": predicted,
                "top_score": top_score,
                "margin": margin,
                "reason": "model_disagreement"
            })

        elif margin < 0.10:

            suspicious_rows.append({
                "customer_text": text,
                "original_intent": actual,
                "model_prediction": predicted,
                "top_score": top_score,
                "margin": margin,
                "reason": "low_margin"
            })

    # ---------------------------------------------------------
    # 6. Create dataframe
    # ---------------------------------------------------------

    suspicious_df = pd.DataFrame(
        suspicious_rows
    )

    if suspicious_df.empty:

        print("\nNo suspicious examples found.")

        return

    # Smaller margin = more ambiguous
    suspicious_df = suspicious_df.sort_values(
        by="margin",
        ascending=True
    )

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    os.makedirs("outputs", exist_ok=True)

    suspicious_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 8. Print summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUSPICIOUS EXAMPLE SUMMARY")
    print("=" * 80)

    print(
        "\nTotal suspicious examples:",
        len(suspicious_df)
    )

    print(
        "\nModel disagreements:",
        (
            suspicious_df["reason"]
            == "model_disagreement"
        ).sum()
    )

    print(
        "Low-margin examples:",
        (
            suspicious_df["reason"]
            == "low_margin"
        ).sum()
    )

    print("\nTop 20 suspicious examples:")

    for index, row in suspicious_df.head(20).iterrows():

        print("\n" + "-" * 80)

        print("Text:")
        print(row["customer_text"])

        print(
            "\nOriginal intent:",
            row["original_intent"]
        )

        print(
            "Model prediction:",
            row["model_prediction"]
        )

        print(
            "Margin:",
            f"{row['margin']:.4f}"
        )

        print(
            "Reason:",
            row["reason"]
        )

    print("\n" + "=" * 80)

    print("Saved to:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print(
        "These examples are candidates for review."
    )

    print(
        "Do NOT automatically change their labels."
    )

    print(
        "They must be reviewed before modifying the training data."
    )


if __name__ == "__main__":
    main()