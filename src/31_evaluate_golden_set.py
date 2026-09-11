import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

GOLDEN_FILE = "evaluation/golden_set_labeled.csv"

TRAIN_FILE = "data/processed/training_data.csv"


# ============================================================
# 2. LOAD GOLDEN SET
# ============================================================

print("=" * 70)
print("LOADING GOLDEN SET")
print("=" * 70)

golden = pd.read_csv(GOLDEN_FILE)

print("Golden set rows:", len(golden))


# Check required columns

required_columns = [
    "customer_text",
    "intent"
]

for column in required_columns:

    if column not in golden.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


golden["customer_text"] = (
    golden["customer_text"]
    .fillna("")
    .astype(str)
)

golden["intent"] = (
    golden["intent"]
    .fillna("")
    .astype(str)
)


X_golden = golden["customer_text"]
y_golden = golden["intent"]


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)

print("Training rows:", len(train))

print()
print("Training columns:")
print(train.columns.tolist())


# ============================================================
# 4. IDENTIFY TEXT + LABEL COLUMNS
# ============================================================

# Your training data should normally contain
# customer_text and intent.

if "customer_text" not in train.columns:
    raise ValueError(
        "customer_text column not found in training data."
    )

if "intent" not in train.columns:
    raise ValueError(
        "intent column not found in training data."
    )


train["customer_text"] = (
    train["customer_text"]
    .fillna("")
    .astype(str)
)

train["intent"] = (
    train["intent"]
    .fillna("")
    .astype(str)
)


X_train = train["customer_text"]
y_train = train["intent"]


# ============================================================
# 5. SHOW TRAINING DISTRIBUTION
# ============================================================

print()
print("Training intent distribution:")
print(
    y_train.value_counts().sort_index()
)


# ============================================================
# 6. WORD TF-IDF
# ============================================================

print()
print("=" * 70)
print("CREATING WORD TF-IDF FEATURES")
print("=" * 70)

word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000,
    sublinear_tf=True,
    min_df=1
)


X_train_word = word_vectorizer.fit_transform(X_train)

X_golden_word = word_vectorizer.transform(X_golden)


print(
    "Word feature shape:",
    X_train_word.shape
)


# ============================================================
# 7. CHARACTER TF-IDF
# ============================================================

print()
print("=" * 70)
print("CREATING CHARACTER TF-IDF FEATURES")
print("=" * 70)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=30000,
    sublinear_tf=True,
    min_df=1
)


X_train_char = char_vectorizer.fit_transform(X_train)

X_golden_char = char_vectorizer.transform(X_golden)


print(
    "Character feature shape:",
    X_train_char.shape
)


# ============================================================
# 8. COMBINE WORD + CHARACTER FEATURES
# ============================================================

print()
print("=" * 70)
print("COMBINING FEATURES")
print("=" * 70)

from scipy.sparse import hstack

X_train_final = hstack(
    [
        X_train_word,
        X_train_char
    ]
)

X_golden_final = hstack(
    [
        X_golden_word,
        X_golden_char
    ]
)


print(
    "Final training shape:",
    X_train_final.shape
)

print(
    "Final golden shape:",
    X_golden_final.shape
)


# ============================================================
# 9. TRAIN LINEAR SVM
# ============================================================

print()
print("=" * 70)
print("TRAINING LINEAR SVM")
print("=" * 70)

model = LinearSVC(
    C=3.0,
    class_weight="balanced"
)

model.fit(
    X_train_final,
    y_train
)


print("Model trained successfully.")


# ============================================================
# 10. PREDICT GOLDEN SET
# ============================================================

print()
print("=" * 70)
print("PREDICTING GOLDEN SET")
print("=" * 70)

y_pred = model.predict(
    X_golden_final
)


# ============================================================
# 11. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_golden,
    y_pred
)

macro_f1 = f1_score(
    y_golden,
    y_pred,
    average="macro"
)


print()
print("=" * 70)
print("GOLDEN SET RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1 : {macro_f1:.4f}"
)

print(
    f"Macro F1 : {macro_f1 * 100:.2f}%"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_golden,
    y_pred,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

labels = sorted(
    y_golden.unique()
)

cm = confusion_matrix(
    y_golden,
    y_pred,
    labels=labels
)


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

results = golden.copy()

results["predicted_intent"] = y_pred

results["correct"] = (
    results["intent"]
    ==
    results["predicted_intent"]
)


output_file = (
    "evaluation/golden_set_predictions.csv"
)

results.to_csv(
    output_file,
    index=False
)


print()
print("Predictions saved to:")
print(output_file)


# ============================================================
# 15. CORRECT / INCORRECT COUNT
# ============================================================

correct = results["correct"].sum()

incorrect = len(results) - correct


print()
print("=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(
    "Correct predictions:",
    correct
)

print(
    "Incorrect predictions:",
    incorrect
)

print(
    "Total:",
    len(results)
)


# ============================================================
# 16. SHOW INCORRECT EXAMPLES
# ============================================================

wrong = results[
    results["correct"] == False
]


print()
print("=" * 70)
print("INCORRECT EXAMPLES")
print("=" * 70)


if len(wrong) == 0:

    print(
        "No incorrect predictions."
    )

else:

    for index, row in wrong.head(20).iterrows():

        print()
        print("-" * 70)

        print(
            "Customer:",
            row["customer_text"]
        )

        print(
            "Actual:",
            row["intent"]
        )

        print(
            "Predicted:",
            row["predicted_intent"]
        )


# ============================================================
# 17. SAVE METRICS
# ============================================================

metrics = pd.DataFrame(
    {
        "metric": [
            "golden_set_size",
            "accuracy",
            "macro_f1",
            "correct_predictions",
            "incorrect_predictions"
        ],

        "value": [
            len(results),
            accuracy,
            macro_f1,
            correct,
            incorrect
        ]
    }
)


metrics_file = (
    "outputs/golden_set_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


print()
print("Metrics saved to:")
print(metrics_file)


# ============================================================
# 18. CONFUSION MATRIX IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 10)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.tight_layout()

cm_image = (
    "outputs/golden_set_confusion_matrix.png"
)

plt.savefig(
    cm_image,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()
print("Confusion matrix image saved to:")
print(cm_image)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("GOLDEN SET EVALUATION COMPLETE")
print("=" * 70)