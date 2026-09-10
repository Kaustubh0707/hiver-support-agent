import os
import pandas as pd


OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "final_evaluation_summary.csv"
)


def add_result(
    results,
    category,
    metric,
    value,
    note
):
    results.append({
        "category": category,
        "metric": metric,
        "value": value,
        "note": note
    })


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    results = []


    # ==================================================
    # 1. CLASSIFIER METRICS
    # ==================================================

    classifier_file = (
        "outputs/classifier_metrics.csv"
    )

    if os.path.exists(classifier_file):

        classifier_df = pd.read_csv(
            classifier_file
        )

        print("\nClassifier metrics columns:")
        print(list(classifier_df.columns))

        for _, row in classifier_df.iterrows():

            add_result(
                results,
                "Intent Classification",
                row["metric"],
                row["value"],
                "Development evaluation using automatically generated pseudo-labels."
            )

    else:

        print(
            "\nWARNING: classifier_metrics.csv not found."
        )


    # ==================================================
    # 2. BASELINE COMPARISON
    # ==================================================

    baseline_file = (
        "outputs/baseline_comparison.csv"
    )

    if os.path.exists(baseline_file):

        baseline_df = pd.read_csv(
            baseline_file
        )

        print("\nBaseline comparison columns:")
        print(list(baseline_df.columns))

        for _, row in baseline_df.iterrows():

            add_result(
                results,
                "Baseline Comparison",
                row["system"],
                row["accuracy"],
                "Accuracy on the same pseudo-labeled development evaluation split."
            )

            add_result(
                results,
                "Baseline Comparison",
                row["system"] + " macro precision",
                row["macro_precision"],
                "Macro precision."
            )

            add_result(
                results,
                "Baseline Comparison",
                row["system"] + " macro recall",
                row["macro_recall"],
                "Macro recall."
            )

            add_result(
                results,
                "Baseline Comparison",
                row["system"] + " macro F1",
                row["macro_f1"],
                "Macro F1."
            )

    else:

        print(
            "\nWARNING: baseline_comparison.csv not found."
        )


    # ==================================================
    # 3. END-TO-END SUPPORT AGENT
    # ==================================================

    agent_file = (
        "outputs/support_agent_evaluation.csv"
    )

    if os.path.exists(agent_file):

        agent_df = pd.read_csv(
            agent_file
        )

        print("\nSupport agent evaluation columns:")
        print(list(agent_df.columns))

        total = len(agent_df)

        auto_handle = (
            agent_df["action"]
            .eq("AUTO-HANDLE")
            .sum()
        )

        escalate = (
            agent_df["action"]
            .eq("ESCALATE")
            .sum()
        )

        add_result(
            results,
            "End-to-End Agent",
            "Examples evaluated",
            total,
            "AppleSupport conversations."
        )

        add_result(
            results,
            "End-to-End Agent",
            "Auto-handle rate",
            auto_handle / total,
            "Current conservative decision policy."
        )

        add_result(
            results,
            "End-to-End Agent",
            "Escalation rate",
            escalate / total,
            "Current conservative decision policy."
        )


        # ----------------------------------------------
        # Confidence
        # ----------------------------------------------

        if "confidence" in agent_df.columns:

            avg_confidence = (
                agent_df["confidence"]
                .mean()
            )

            add_result(
                results,
                "End-to-End Agent",
                "Average intent confidence",
                avg_confidence,
                "Classifier confidence."
            )


        # ----------------------------------------------
        # Historical similarity
        # ----------------------------------------------

        if "best_similarity" in agent_df.columns:

            avg_similarity = (
                agent_df["best_similarity"]
                .mean()
            )

            strong_evidence = (
                agent_df["best_similarity"]
                .ge(0.50)
                .mean()
            )

            add_result(
                results,
                "End-to-End Agent",
                "Average historical similarity",
                avg_similarity,
                "Best retrieved historical example."
            )

            add_result(
                results,
                "End-to-End Agent",
                "Examples with strong evidence",
                strong_evidence,
                "Best historical similarity >= 0.50."
            )


        # ----------------------------------------------
        # Evidence count, if available
        # ----------------------------------------------

        if "evidence_count" in agent_df.columns:

            evidence_available = (
                agent_df["evidence_count"]
                .gt(0)
                .mean()
            )

            add_result(
                results,
                "End-to-End Agent",
                "Examples with evidence",
                evidence_available,
                "At least one historical example retrieved."
            )


        # ----------------------------------------------
        # Escalation reasons
        # ----------------------------------------------

        if "decision_reason" in agent_df.columns:

            print("\nEscalation reasons:")
            print("-" * 60)

            reasons = (
                agent_df[
                    agent_df["action"]
                    == "ESCALATE"
                ]["decision_reason"]
                .value_counts()
            )

            print(reasons)

            for reason, count in reasons.items():

                add_result(
                    results,
                    "Escalation Analysis",
                    reason,
                    count,
                    "Number of escalated examples."
                )

    else:

        print(
            "\nWARNING: support_agent_evaluation.csv not found."
        )


    # ==================================================
    # 4. THRESHOLD TUNING
    # ==================================================

    # (Old One)threshold_file = (
    #     "outputs/threshold_tuning.csv"
    # )

    threshold_file = "outputs/decision_threshold_comparison.csv"

    if os.path.exists(threshold_file):

        threshold_df = pd.read_csv(
            threshold_file
        )

        print("\nThreshold tuning columns:")
        print(list(threshold_df.columns))

        for _, row in threshold_df.iterrows():

            policy = str(
                row.iloc[0]
            )

            if "auto_handle_rate" in threshold_df.columns:

                value = row[
                    "auto_handle_rate"
                ]

            else:

                value = row.iloc[1]

            add_result(
                results,
                "Decision Policy",
                policy + " auto-handle rate",
                value,
                "Threshold sensitivity analysis."
            )

    else:

        print(
            # (Old One) "\nWARNING: threshold_tuning.csv not found."
            "\nWARNING: decision_threshold_comparison.csv not found."
        )


    # ==================================================
    # 5. REPLY QUALITY LIMITATION
    # ==================================================

    add_result(
        results,
        "Reply Quality",
        "Local proxy score",
        7.0,
        (
            "NOT used as a headline metric. "
            "All 500 examples received the same score, "
            "showing that the rule-based proxy is not "
            "discriminative enough."
        )
    )


    # ==================================================
    # 6. EVALUATION LIMITATIONS
    # ==================================================

    add_result(
        results,
        "Evaluation Limitation",
        "Hand-labeled golden set",
        0,
        (
            "Required 150-250 hand-labeled examples "
            "were not completed."
        )
    )

    add_result(
        results,
        "Evaluation Limitation",
        "LLM-as-judge",
        0,
        (
            "Not performed because no external LLM API "
            "was used."
        )
    )


    # ==================================================
    # 7. SAVE
    # ==================================================

    final_df = pd.DataFrame(
        results
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ==================================================
    # 8. PRINT SUMMARY
    # ==================================================

    print("\n")
    print("=" * 80)
    print("FINAL HEADLINE RESULTS")
    print("=" * 80)


    print("\nIntent Classification:")

    print(
        final_df[
            final_df["category"]
            == "Intent Classification"
        ].to_string(index=False)
    )


    print("\nBaseline Comparison:")

    print(
        final_df[
            final_df["category"]
            == "Baseline Comparison"
        ].to_string(index=False)
    )


    print("\nEnd-to-End Agent:")

    print(
        final_df[
            final_df["category"]
            == "End-to-End Agent"
        ].to_string(index=False)
    )


    print("\n")
    print("=" * 80)
    print("IMPORTANT INTERPRETATION")
    print("=" * 80)

    print(
        "\n1. The 79.1% classifier accuracy is "
        "a development result based on pseudo-labels."
    )

    print(
        "\n2. It must NOT be presented as "
        "human-validated intent accuracy."
    )

    print(
        "\n3. The 7/9 local reply-quality score "
        "is not a reliable headline metric because "
        "all 500 examples received the same score."
    )

    print(
        "\n4. The conservative decision policy "
        "intentionally escalates uncertain or risky cases."
    )

    print(
        "\n5. The required hand-labeled golden set "
        "and LLM-as-judge evaluation were not completed."
    )


    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION SUMMARY SAVED")
    print("=" * 80)

    print("\nFile:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()