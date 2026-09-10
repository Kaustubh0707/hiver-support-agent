import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


INPUT_FILE = "data/processed/training_data.csv"

MODEL_FOLDER = "models"
MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "intent_classifier.joblib"
)


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("TRAINING INTENT CLASSIFIER")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Check input file
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Training data not found.")

        print("\nExpected file:")
        print(INPUT_FILE)

        print("\nPlease run:")
        print("python src/07_create_training_data.py")

        return


    # ---------------------------------------------------------
    # 2. Load training data
    # ---------------------------------------------------------

    print("\nLoading training data...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Total examples:", len(df))


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
        (df["customer_text"] != "")
        & (df["intent"] != "")
    ].copy()

    print("Usable examples:", len(df))


    # ---------------------------------------------------------
    # 4. Separate input and target
    # ---------------------------------------------------------

    X = df["customer_text"]

    y = df["intent"]


    # ---------------------------------------------------------
    # 5. Train / validation split
    # ---------------------------------------------------------

    print("\nCreating train/validation split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training examples:", len(X_train))
    print("Validation examples:", len(X_test))


    # ---------------------------------------------------------
    # 6. Build TF-IDF + Logistic Regression pipeline
    # ---------------------------------------------------------

    print("\nBuilding TF-IDF + Logistic Regression model...")

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


    # ---------------------------------------------------------
    # 7. Train model
    # ---------------------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")


    # ---------------------------------------------------------
    # 8. Make predictions
    # ---------------------------------------------------------

    print("\nEvaluating model...")

    predictions = model.predict(X_test)


    # ---------------------------------------------------------
    # 9. Calculate accuracy
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 80)
    print("CLASSIFIER RESULTS")
    print("=" * 80)

    print("\nAccuracy:", round(accuracy, 4))


    # ---------------------------------------------------------
    # 10. Classification report
    # ---------------------------------------------------------

    print("\nClassification Report")
    print("-" * 80)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


    # ---------------------------------------------------------
    # 11. Save model
    # ---------------------------------------------------------

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\nModel saved successfully.")

    print("\nModel file:")
    print(MODEL_FILE)


    # ---------------------------------------------------------
    # 12. Test a few messages
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)

    sample_messages = [
        "My iPhone battery is draining very quickly.",
        "My phone keeps restarting after the update.",
        "How do I update my iPhone?",
        "Bluetooth is not connecting.",
        "iTunes is not working.",
        "My keyboard stopped working."
    ]

    for message in sample_messages:

        prediction = model.predict(
            [message]
        )[0]

        probabilities = model.predict_proba(
            [message]
        )[0]

        confidence = probabilities.max()

        print("\nCustomer:")
        print(message)

        print("Predicted intent:")
        print(prediction)

        print("Confidence:")
        print(round(confidence, 4))


if __name__ == "__main__":
    main()