# (Old One)

# import os
# import pandas as pd


# OUTPUT_FOLDER = "outputs"
# OUTPUT_FILE = os.path.join(
#     OUTPUT_FOLDER,
#     "final_evaluation_summary.csv"
# )


# def add_result(
#     results,
#     category,
#     metric,
#     value,
#     note
# ):
#     results.append({
#         "category": category,
#         "metric": metric,
#         "value": value,
#         "note": note
#     })


# def main():

#     print("=" * 80)
#     print("HIVER AI SUPPORT AGENT")
#     print("FINAL EVALUATION SUMMARY")
#     print("=" * 80)

#     os.makedirs(
#         OUTPUT_FOLDER,
#         exist_ok=True
#     )

#     results = []


#     # ==================================================
#     # 1. CLASSIFIER METRICS
#     # ==================================================

#     classifier_file = (
#         "outputs/classifier_metrics.csv"
#     )

#     if os.path.exists(classifier_file):

#         classifier_df = pd.read_csv(
#             classifier_file
#         )

#         print("\nClassifier metrics columns:")
#         print(list(classifier_df.columns))

#         for _, row in classifier_df.iterrows():

#             add_result(
#                 results,
#                 "Intent Classification",
#                 row["metric"],
#                 row["value"],
#                 "Development evaluation using automatically generated pseudo-labels."
#             )

#     else:

#         print(
#             "\nWARNING: classifier_metrics.csv not found."
#         )


#     # ==================================================
#     # 2. BASELINE COMPARISON
#     # ==================================================

#     baseline_file = (
#         "outputs/baseline_comparison.csv"
#     )

#     if os.path.exists(baseline_file):

#         baseline_df = pd.read_csv(
#             baseline_file
#         )

#         print("\nBaseline comparison columns:")
#         print(list(baseline_df.columns))

#         for _, row in baseline_df.iterrows():

#             add_result(
#                 results,
#                 "Baseline Comparison",
#                 row["system"],
#                 row["accuracy"],
#                 "Accuracy on the same pseudo-labeled development evaluation split."
#             )

#             add_result(
#                 results,
#                 "Baseline Comparison",
#                 row["system"] + " macro precision",
#                 row["macro_precision"],
#                 "Macro precision."
#             )

#             add_result(
#                 results,
#                 "Baseline Comparison",
#                 row["system"] + " macro recall",
#                 row["macro_recall"],
#                 "Macro recall."
#             )

#             add_result(
#                 results,
#                 "Baseline Comparison",
#                 row["system"] + " macro F1",
#                 row["macro_f1"],
#                 "Macro F1."
#             )

#     else:

#         print(
#             "\nWARNING: baseline_comparison.csv not found."
#         )


#     # ==================================================
#     # 3. END-TO-END SUPPORT AGENT
#     # ==================================================

#     agent_file = (
#         "outputs/support_agent_evaluation.csv"
#     )

#     if os.path.exists(agent_file):

#         agent_df = pd.read_csv(
#             agent_file
#         )

#         print("\nSupport agent evaluation columns:")
#         print(list(agent_df.columns))

#         total = len(agent_df)

#         auto_handle = (
#             agent_df["action"]
#             .eq("AUTO-HANDLE")
#             .sum()
#         )

#         escalate = (
#             agent_df["action"]
#             .eq("ESCALATE")
#             .sum()
#         )

#         add_result(
#             results,
#             "End-to-End Agent",
#             "Examples evaluated",
#             total,
#             "AppleSupport conversations."
#         )

#         add_result(
#             results,
#             "End-to-End Agent",
#             "Auto-handle rate",
#             auto_handle / total,
#             "Current conservative decision policy."
#         )

#         add_result(
#             results,
#             "End-to-End Agent",
#             "Escalation rate",
#             escalate / total,
#             "Current conservative decision policy."
#         )


#         # ----------------------------------------------
#         # Confidence
#         # ----------------------------------------------

#         if "confidence" in agent_df.columns:

#             avg_confidence = (
#                 agent_df["confidence"]
#                 .mean()
#             )

#             add_result(
#                 results,
#                 "End-to-End Agent",
#                 "Average intent confidence",
#                 avg_confidence,
#                 "Classifier confidence."
#             )


#         # ----------------------------------------------
#         # Historical similarity
#         # ----------------------------------------------

#         if "best_similarity" in agent_df.columns:

#             avg_similarity = (
#                 agent_df["best_similarity"]
#                 .mean()
#             )

#             strong_evidence = (
#                 agent_df["best_similarity"]
#                 .ge(0.50)
#                 .mean()
#             )

#             add_result(
#                 results,
#                 "End-to-End Agent",
#                 "Average historical similarity",
#                 avg_similarity,
#                 "Best retrieved historical example."
#             )

#             add_result(
#                 results,
#                 "End-to-End Agent",
#                 "Examples with strong evidence",
#                 strong_evidence,
#                 "Best historical similarity >= 0.50."
#             )


#         # ----------------------------------------------
#         # Evidence count, if available
#         # ----------------------------------------------

#         if "evidence_count" in agent_df.columns:

#             evidence_available = (
#                 agent_df["evidence_count"]
#                 .gt(0)
#                 .mean()
#             )

#             add_result(
#                 results,
#                 "End-to-End Agent",
#                 "Examples with evidence",
#                 evidence_available,
#                 "At least one historical example retrieved."
#             )


#         # ----------------------------------------------
#         # Escalation reasons
#         # ----------------------------------------------

#         if "decision_reason" in agent_df.columns:

#             print("\nEscalation reasons:")
#             print("-" * 60)

#             reasons = (
#                 agent_df[
#                     agent_df["action"]
#                     == "ESCALATE"
#                 ]["decision_reason"]
#                 .value_counts()
#             )

#             print(reasons)

#             for reason, count in reasons.items():

#                 add_result(
#                     results,
#                     "Escalation Analysis",
#                     reason,
#                     count,
#                     "Number of escalated examples."
#                 )

#     else:

#         print(
#             "\nWARNING: support_agent_evaluation.csv not found."
#         )


#     # ==================================================
#     # 4. THRESHOLD TUNING
#     # ==================================================

#     # (Old One)threshold_file = (
#     #     "outputs/threshold_tuning.csv"
#     # )

#     threshold_file = "outputs/decision_threshold_comparison.csv"

#     if os.path.exists(threshold_file):

#         threshold_df = pd.read_csv(
#             threshold_file
#         )

#         print("\nThreshold tuning columns:")
#         print(list(threshold_df.columns))

#         for _, row in threshold_df.iterrows():

#             policy = str(
#                 row.iloc[0]
#             )

#             if "auto_handle_rate" in threshold_df.columns:

#                 value = row[
#                     "auto_handle_rate"
#                 ]

#             else:

#                 value = row.iloc[1]

#             add_result(
#                 results,
#                 "Decision Policy",
#                 policy + " auto-handle rate",
#                 value,
#                 "Threshold sensitivity analysis."
#             )

#     else:

#         print(
#             # (Old One) "\nWARNING: threshold_tuning.csv not found."
#             "\nWARNING: decision_threshold_comparison.csv not found."
#         )


#     # ==================================================
#     # 5. REPLY QUALITY LIMITATION
#     # ==================================================

#     add_result(
#         results,
#         "Reply Quality",
#         "Local proxy score",
#         7.0,
#         (
#             "NOT used as a headline metric. "
#             "All 500 examples received the same score, "
#             "showing that the rule-based proxy is not "
#             "discriminative enough."
#         )
#     )


#     # ==================================================
#     # 6. EVALUATION LIMITATIONS
#     # ==================================================

#     add_result(
#         results,
#         "Evaluation Limitation",
#         "Hand-labeled golden set",
#         0,
#         (
#             "Required 150-250 hand-labeled examples "
#             "were not completed."
#         )
#     )

#     add_result(
#         results,
#         "Evaluation Limitation",
#         "LLM-as-judge",
#         0,
#         (
#             "Not performed because no external LLM API "
#             "was used."
#         )
#     )


#     # ==================================================
#     # 7. SAVE
#     # ==================================================

#     final_df = pd.DataFrame(
#         results
#     )

#     final_df.to_csv(
#         OUTPUT_FILE,
#         index=False
#     )


#     # ==================================================
#     # 8. PRINT SUMMARY
#     # ==================================================

#     print("\n")
#     print("=" * 80)
#     print("FINAL HEADLINE RESULTS")
#     print("=" * 80)


#     print("\nIntent Classification:")

#     print(
#         final_df[
#             final_df["category"]
#             == "Intent Classification"
#         ].to_string(index=False)
#     )


#     print("\nBaseline Comparison:")

#     print(
#         final_df[
#             final_df["category"]
#             == "Baseline Comparison"
#         ].to_string(index=False)
#     )


#     print("\nEnd-to-End Agent:")

#     print(
#         final_df[
#             final_df["category"]
#             == "End-to-End Agent"
#         ].to_string(index=False)
#     )


#     print("\n")
#     print("=" * 80)
#     print("IMPORTANT INTERPRETATION")
#     print("=" * 80)

#     print(
#         "\n1. The 79.1% classifier accuracy is "
#         "a development result based on pseudo-labels."
#     )

#     print(
#         "\n2. It must NOT be presented as "
#         "human-validated intent accuracy."
#     )

#     print(
#         "\n3. The 7/9 local reply-quality score "
#         "is not a reliable headline metric because "
#         "all 500 examples received the same score."
#     )

#     print(
#         "\n4. The conservative decision policy "
#         "intentionally escalates uncertain or risky cases."
#     )

#     print(
#         "\n5. The required hand-labeled golden set "
#         "and LLM-as-judge evaluation were not completed."
#     )


#     print("\n")
#     print("=" * 80)
#     print("FINAL EVALUATION SUMMARY SAVED")
#     print("=" * 80)

#     print("\nFile:")
#     print(OUTPUT_FILE)


# if __name__ == "__main__":
#     main()

# (New One) 

import os
import pandas as pd


# ---------------------------------------------------------
# Input files
# ---------------------------------------------------------

GOLDEN_FILE = "outputs/golden_set_metrics.csv"
E2E_FILE = "outputs/end_to_end_metrics.csv"
SVM_FILE = "outputs/svm_tuning_results.csv"

OUTPUT_FILE = "outputs/final_evaluation_summary.csv"


def get_metric(df, metric_name):
    """Return value for an exact metric name."""
    row = df[df["metric"].astype(str).str.lower() == metric_name.lower()]

    if len(row) == 0:
        return None

    return row.iloc[0]["value"]


def main():

    rows = []

    # =====================================================
    # 1. GOLDEN SET
    # =====================================================

    golden = pd.read_csv(GOLDEN_FILE)

    golden_size = get_metric(
        golden,
        "golden_set_size"
    )

    golden_accuracy = get_metric(
        golden,
        "accuracy"
    )

    golden_macro_f1 = get_metric(
        golden,
        "macro_f1"
    )

    rows.append({
        "metric": "Reviewed/provisional golden-set size",
        "value": golden_size,
        "notes": (
            "200 AppleSupport customer messages. "
            "Labels were reviewed during development but were not "
            "independently double-annotated."
        )
    })

    rows.append({
        "metric": "Golden-set accuracy",
        "value": golden_accuracy,
        "notes": (
            "Evaluation of the Word + Character TF-IDF + LinearSVC "
            "classifier on the reviewed/provisional set."
        )
    })

    rows.append({
        "metric": "Golden-set macro F1",
        "value": golden_macro_f1,
        "notes": (
            "Macro F1 across the 10 intent classes."
        )
    })

    # =====================================================
    # 2. DEVELOPMENT / PSEUDO-LABEL BENCHMARK
    # =====================================================

    svm = pd.read_csv(SVM_FILE)

    best_row = svm.sort_values(
        "Macro_F1",
        ascending=False
    ).iloc[0]

    best_c = best_row["C"]
    best_accuracy = best_row["Accuracy"]
    best_macro_f1 = best_row["Macro_F1"]

    rows.append({
        "metric": "Best pseudo-label benchmark accuracy",
        "value": best_accuracy,
        "notes": (
            f"Word + Character TF-IDF + LinearSVC with C={best_c}. "
            "Labels were generated using keyword-based rules, so this "
            "is not a human-validated accuracy estimate."
        )
    })

    rows.append({
        "metric": "Best pseudo-label benchmark macro F1",
        "value": best_macro_f1,
        "notes": (
            f"Word + Character TF-IDF + LinearSVC with C={best_c}. "
            "Pseudo-label benchmark."
        )
    })

    # Baselines
    rows.append({
        "metric": "Majority-class baseline accuracy",
        "value": 0.10,
        "notes": (
            "10 intent classes with approximately balanced pseudo-label "
            "training data."
        )
    })

    rows.append({
        "metric": "Keyword-rule baseline accuracy",
        "value": 0.9560,
        "notes": (
            "Keyword baseline. This is artificially strong because the "
            "pseudo-labels were themselves generated using keyword rules."
        )
    })

    # =====================================================
    # 3. END-TO-END
    # =====================================================

    e2e = pd.read_csv(E2E_FILE)

    e2e_metrics = {
        row["metric"]: row["value"]
        for _, row in e2e.iterrows()
    }

    rows.extend([
        {
            "metric": "End-to-end evaluation examples",
            "value": e2e_metrics.get(
                "Evaluation examples"
            ),
            "notes": "AppleSupport historical examples."
        },
        {
            "metric": "Auto-handle rate",
            "value": e2e_metrics.get(
                "AUTO-HANDLE rate"
            ),
            "notes": (
                "Final policy using intent confidence, retrieval "
                "similarity, and safety/escalation rules."
            )
        },
        {
            "metric": "Escalation rate",
            "value": e2e_metrics.get(
                "Escalation rate"
            ),
            "notes": (
                "Cases not considered safe enough for automatic handling."
            )
        },
        {
            "metric": "Average intent confidence",
            "value": e2e_metrics.get(
                "Average intent confidence"
            ),
            "notes": "Mean classifier confidence over 500 examples."
        },
        {
            "metric": "Evidence availability rate",
            "value": e2e_metrics.get(
                "Evidence availability rate"
            ),
            "notes": (
                "Fraction of examples for which historical retrieval "
                "returned evidence."
            )
        },
        {
            "metric": "Strong evidence rate",
            "value": e2e_metrics.get(
                "Strong evidence rate"
            ),
            "notes": (
                "Historical retrieval similarity >= 0.50."
            )
        },
        {
            "metric": "Weak auto-handle rate",
            "value": e2e_metrics.get(
                "Weak AUTO-HANDLE rate"
            ),
            "notes": (
                "Fraction of auto-handled cases classified as weak "
                "according to the evaluation's evidence criteria."
            )
        }
    ])

    # =====================================================
    # 4. LLM-AS-JUDGE
    # =====================================================

    rows.append({
        "metric": "LLM-as-judge",
        "value": "Not executed",
        "notes": (
            "No external LLM API key was available in the development "
            "environment. No fabricated judge scores are reported."
        )
    })

    rows.append({
        "metric": "Human/LLM agreement",
        "value": "Not measured",
        "notes": (
            "Because an LLM judge was not executed, human/LLM agreement "
            "cannot be honestly calculated."
        )
    })

    # =====================================================
    # 5. REQUIRED MISLEADING-HEADLINE ANALYSIS
    # =====================================================

    rows.append({
        "metric": "What is misleading about my headline number?",
        "value": "91.30%",
        "notes": (
            "The 91.30% classifier result is measured against pseudo-labels "
            "created by keyword rules. The same model achieves 23.00% "
            "accuracy and 19.30% macro F1 on the 200-example "
            "reviewed/provisional golden set. Therefore 91.30% should "
            "not be presented as human-validated real-world accuracy."
        )
    })

    # =====================================================
    # 6. SAFETY / ESCALATION
    # =====================================================

    rows.append({
        "metric": "Safety philosophy",
        "value": "Conservative auto-handling",
        "notes": (
            "Sensitive, unresolved, account/billing, device-specific, "
            "low-confidence, low-similarity, and ambiguous cases are "
            "preferentially escalated."
        )
    })

    # =====================================================
    # SAVE
    # =====================================================

    output = pd.DataFrame(rows)

    os.makedirs("outputs", exist_ok=True)

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print()
    print(output.to_string(index=False))
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()