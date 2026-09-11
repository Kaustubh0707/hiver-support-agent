import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


INPUT_FILE = "data/processed/training_data.csv"


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("SVM HYPERPARAMETER TUNING")
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

    print("Examples after cleaning:", len(df))

    # ---------------------------------------------------------
    # 2. Train/test split
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
    # 3. Feature extraction
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
    # 4. Values to test
    # ---------------------------------------------------------

    c_values = [
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0
    ]

    results = []

    # ---------------------------------------------------------
    # 5. Train models
    # ---------------------------------------------------------

    for c_value in c_values:

        print("\n" + "-" * 80)
        print(f"Testing C = {c_value}")
        print("-" * 80)

        model = Pipeline([
            (
                "features",
                features
            ),
            (
                "classifier",
                LinearSVC(
                    C=c_value,
                    class_weight="balanced"
                )
            )
        ])

        print("Training...")

        model.fit(X_train, y_train)

        print("Predicting...")

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        results.append({
            "C": c_value,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1
        })

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Macro F1 : {macro_f1:.4f}")

    # ---------------------------------------------------------
    # 6. Results table
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Macro_F1",
        ascending=False
    )

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print()

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Macro_F1": "{:.4f}".format
            }
        )
    )

    # ---------------------------------------------------------
    # 7. Best model
    # ---------------------------------------------------------

    best = results_df.iloc[0]

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)

    print(f"\nBest C       : {best['C']}")
    print(f"Accuracy     : {best['Accuracy']:.4f}")
    print(f"Macro F1     : {best['Macro_F1']:.4f}")

    improvement = (
        best["Accuracy"] - 0.911
    ) * 100

    print(
        f"\nImprovement over current 91.10%: "
        f"{improvement:+.2f} percentage points"
    )

    # ---------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------

    output_file = "outputs/svm_tuning_results.csv"

    os.makedirs("outputs", exist_ok=True)

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)

    # ---------------------------------------------------------
    # 9. Important note
    # ---------------------------------------------------------

    print("\nIMPORTANT:")
    print(
        "These results use automatically generated pseudo-labels."
    )

    print(
        "The best configuration must be validated later "
        "on the human-labelled golden set."
    )


if __name__ == "__main__":
    main()