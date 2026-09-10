import os
import pandas as pd

INPUT_FILE = "outputs/support_agent_evaluation.csv"
OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "decision_threshold_comparison.csv"
)


def evaluate_policy(df, policy_name, confidence_threshold, similarity_threshold):

    eligible = (
        (df["confidence"] >= confidence_threshold)
        &
        (df["best_similarity"] >= similarity_threshold)
        &
        (~df["predicted_intent"].isin([
            "general_support",
            "other",
            "performance_freezing",
            "device_functionality_issue"
        ]))
    )

    auto_count = eligible.sum()
    total = len(df)

    auto_rate = auto_count / total

    selected = df[eligible]

    if len(selected) > 0:
        avg_confidence = selected["confidence"].mean()
        avg_similarity = selected["best_similarity"].mean()
    else:
        avg_confidence = 0
        avg_similarity = 0

    return {
        "policy": policy_name,
        "confidence_threshold": confidence_threshold,
        "similarity_threshold": similarity_threshold,
        "auto_handle_count": int(auto_count),
        "escalate_count": int(total - auto_count),
        "auto_handle_rate": round(auto_rate, 4),
        "escalation_rate": round(1 - auto_rate, 4),
        "average_auto_handle_confidence": round(
            avg_confidence, 4
        ),
        "average_auto_handle_similarity": round(
            avg_similarity, 4
        )
    }


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("DECISION THRESHOLD TUNING")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Evaluation file not found.")

        print("\nExpected:")
        print(INPUT_FILE)

        print("\nPlease run:")
        print("python src/15_evaluate_support_agent.py")

        return

    print("\nLoading evaluation results...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Evaluation examples:", len(df))

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    df["best_similarity"] = pd.to_numeric(
        df["best_similarity"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "confidence",
            "best_similarity"
        ]
    ).copy()

    print("Usable examples:", len(df))

    # ---------------------------------------------------------
    # POLICIES
    # ---------------------------------------------------------

    policies = [
        (
            "Current",
            0.60,
            0.40
        ),
        (
            "Balanced",
            0.70,
            0.50
        ),
        (
            "Conservative",
            0.80,
            0.60
        ),
        (
            "Very Conservative",
            0.90,
            0.70
        )
    ]

    results = []

    print("\nTesting policies...")

    for policy_name, confidence_threshold, similarity_threshold in policies:

        result = evaluate_policy(
            df,
            policy_name,
            confidence_threshold,
            similarity_threshold
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("POLICY COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(index=False)
    )

    print("\n")
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    for _, row in results_df.iterrows():

        print(
            f"\n{row['policy']} Policy:"
        )

        print(
            f"  Confidence >= "
            f"{row['confidence_threshold']}"
        )

        print(
            f"  Similarity >= "
            f"{row['similarity_threshold']}"
        )

        print(
            f"  AUTO-HANDLE: "
            f"{row['auto_handle_count']} / "
            f"{len(df)} "
            f"({row['auto_handle_rate'] * 100:.1f}%)"
        )

        print(
            f"  ESCALATE: "
            f"{row['escalate_count']} / "
            f"{len(df)} "
            f"({row['escalation_rate'] * 100:.1f}%)"
        )

        print(
            f"  Avg AUTO-HANDLE confidence: "
            f"{row['average_auto_handle_confidence']}"
        )

        print(
            f"  Avg AUTO-HANDLE similarity: "
            f"{row['average_auto_handle_similarity']}"
        )

    print("\n")
    print("=" * 80)
    print("RECOMMENDED POLICY")
    print("=" * 80)

    print("""
The Balanced policy is the intended production trade-off.

It requires:

- Intent confidence >= 0.70
- Historical similarity >= 0.50
- No broad or high-risk intent
- No device-specific troubleshooting

This favors evidence quality over maximum automation coverage.
""")

    print("\nSaved file:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("THRESHOLD TUNING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()