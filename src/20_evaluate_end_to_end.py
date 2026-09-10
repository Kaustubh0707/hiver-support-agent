import os
import pandas as pd

INPUT_FILE = "outputs/support_agent_evaluation.csv"
OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "end_to_end_metrics.csv"
)


def main():
    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("END-TO-END EVALUATION")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Evaluation file not found.")
        print("\nExpected:")
        print(INPUT_FILE)
        print("\nPlease run:")
        print("python src/15_evaluate_support_agent.py")
        return

    print("\nLoading support-agent evaluation...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("Evaluation examples:", len(df))

    # Convert numeric columns safely
    numeric_columns = [
        "confidence",
        "best_similarity",
        "retrieved_count"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    total = len(df)

    if total == 0:
        print("\nERROR: No evaluation examples found.")
        return

    # ---------------------------------------------------------
    # BASIC METRICS
    # ---------------------------------------------------------

    auto_handle_count = (
        df["action"] == "AUTO-HANDLE"
    ).sum()

    escalate_count = (
        df["action"] == "ESCALATE"
    ).sum()

    auto_handle_rate = auto_handle_count / total
    escalation_rate = escalate_count / total

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    average_confidence = df["confidence"].mean()

    high_confidence_count = (
        df["confidence"] >= 0.70
    ).sum()

    high_confidence_rate = (
        high_confidence_count / total
    )

    # ---------------------------------------------------------
    # RETRIEVAL EVIDENCE
    # ---------------------------------------------------------

    evidence_available_count = (
        df["retrieved_count"] > 0
    ).sum()

    evidence_available_rate = (
        evidence_available_count / total
    )

    average_similarity = (
        df["best_similarity"].fillna(0).mean()
    )

    strong_evidence_count = (
        df["best_similarity"] >= 0.50
    ).sum()

    strong_evidence_rate = (
        strong_evidence_count / total
    )

    # ---------------------------------------------------------
    # AUTOMATION QUALITY SIGNALS
    # ---------------------------------------------------------

    auto_handle_df = df[
        df["action"] == "AUTO-HANDLE"
    ].copy()

    if len(auto_handle_df) > 0:

        auto_avg_confidence = (
            auto_handle_df["confidence"].mean()
        )

        auto_avg_similarity = (
            auto_handle_df["best_similarity"]
            .fillna(0)
            .mean()
        )

        weak_auto_count = (
            (
                auto_handle_df["confidence"] < 0.70
            )
            |
            (
                auto_handle_df["best_similarity"] < 0.50
            )
        ).sum()

        weak_auto_rate = (
            weak_auto_count / len(auto_handle_df)
        )

    else:

        auto_avg_confidence = 0
        auto_avg_similarity = 0
        weak_auto_count = 0
        weak_auto_rate = 0

    # ---------------------------------------------------------
    # ESCALATION REASONS
    # ---------------------------------------------------------

    escalation_df = df[
        df["action"] == "ESCALATE"
    ]

    escalation_reason_counts = (
        escalation_df["decision_reason"]
        .value_counts()
    )

    # ---------------------------------------------------------
    # INTENT DISTRIBUTION
    # ---------------------------------------------------------

    intent_distribution = (
        df["predicted_intent"]
        .value_counts()
    )

    # ---------------------------------------------------------
    # CREATE METRICS TABLE
    # ---------------------------------------------------------

    metrics = [
        {
            "metric": "Evaluation examples",
            "value": total
        },
        {
            "metric": "AUTO-HANDLE count",
            "value": auto_handle_count
        },
        {
            "metric": "ESCALATE count",
            "value": escalate_count
        },
        {
            "metric": "AUTO-HANDLE rate",
            "value": round(auto_handle_rate, 4)
        },
        {
            "metric": "Escalation rate",
            "value": round(escalation_rate, 4)
        },
        {
            "metric": "Average intent confidence",
            "value": round(average_confidence, 4)
        },
        {
            "metric": "High-confidence predictions (>= 0.70)",
            "value": high_confidence_count
        },
        {
            "metric": "High-confidence prediction rate",
            "value": round(high_confidence_rate, 4)
        },
        {
            "metric": "Cases with retrieval evidence",
            "value": evidence_available_count
        },
        {
            "metric": "Evidence availability rate",
            "value": round(evidence_available_rate, 4)
        },
        {
            "metric": "Average historical similarity",
            "value": round(average_similarity, 4)
        },
        {
            "metric": "Cases with strong evidence (similarity >= 0.50)",
            "value": strong_evidence_count
        },
        {
            "metric": "Strong evidence rate",
            "value": round(strong_evidence_rate, 4)
        },
        {
            "metric": "AUTO-HANDLE average confidence",
            "value": round(auto_avg_confidence, 4)
        },
        {
            "metric": "AUTO-HANDLE average similarity",
            "value": round(auto_avg_similarity, 4)
        },
        {
            "metric": "Weak AUTO-HANDLE cases",
            "value": weak_auto_count
        },
        {
            "metric": "Weak AUTO-HANDLE rate",
            "value": round(weak_auto_rate, 4)
        }
    ]

    metrics_df = pd.DataFrame(metrics)

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    metrics_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("END-TO-END METRICS")
    print("=" * 80)

    print(
        metrics_df.to_string(index=False)
    )

    print("\n")
    print("=" * 80)
    print("ESCALATION REASONS")
    print("=" * 80)

    if len(escalation_reason_counts) > 0:

        for reason, count in escalation_reason_counts.items():

            percentage = count / total * 100

            print(
                f"{reason}: "
                f"{count} "
                f"({percentage:.1f}%)"
            )

    print("\n")
    print("=" * 80)
    print("PREDICTED INTENT DISTRIBUTION")
    print("=" * 80)

    for intent, count in intent_distribution.items():

        percentage = count / total * 100

        print(
            f"{intent}: "
            f"{count} "
            f"({percentage:.1f}%)"
        )

    print("\n")
    print("=" * 80)
    print("IMPORTANT EVALUATION NOTE")
    print("=" * 80)

    print("""
These metrics evaluate the current local support-agent pipeline.

The intent classifier was trained and evaluated using
keyword-generated pseudo-labels.

Therefore:

- These results are development metrics.
- They are NOT equivalent to accuracy on a hand-labelled
  golden evaluation set.
- No human/LLM agreement is claimed.
- The assignment-required 150-250 hand-labelled examples
  have not been completed.
""")

    print("\nSaved metrics:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("END-TO-END EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()