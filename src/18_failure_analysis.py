import os
import pandas as pd


INPUT_FILE = "outputs/support_agent_evaluation.csv"

OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "top_failure_cases.csv"
)


def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("FAILURE ANALYSIS")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Evaluation file not found.")

        print(
            "\nExpected:"
        )

        print(
            INPUT_FILE
        )

        print(
            "\nPlease run Step 15 first."
        )

        return

    print(
        "\nLoading evaluation results..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        "Evaluation examples:",
        len(df)
    )

    # -----------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    df["best_similarity"] = pd.to_numeric(
        df["best_similarity"],
        errors="coerce"
    )

    df["retrieved_count"] = pd.to_numeric(
        df["retrieved_count"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Create failure score
    # -----------------------------------------------------

    # Lower confidence = higher risk
    confidence_risk = (
        1 - df["confidence"]
    )

    # Lower similarity = higher risk
    similarity_risk = (
        1 - df["best_similarity"].clip(
            lower=0,
            upper=1
        )
    )

    # No evidence = high risk
    evidence_risk = (
        df["retrieved_count"] == 0
    ).astype(int)

    # Auto-handling with weak evidence
    auto_handle_risk = (
        (
            df["action"] == "AUTO-HANDLE"
        )
        &
        (
            (
                df["confidence"] < 0.70
            )
            |
            (
                df["best_similarity"] < 0.50
            )
        )
    ).astype(int)

    # -----------------------------------------------------
    # Combined risk score
    # -----------------------------------------------------

    df["failure_score"] = (

        confidence_risk * 0.40

        +

        similarity_risk * 0.40

        +

        evidence_risk * 0.10

        +

        auto_handle_risk * 0.10
    )

    # -----------------------------------------------------
    # Sort worst cases first
    # -----------------------------------------------------

    failures = df.sort_values(
        "failure_score",
        ascending=False
    ).copy()

    top_failures = failures.head(
        5
    ).copy()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    top_failures.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TOP 5 FAILURE CASES")
    print("=" * 80)

    for number, (_, row) in enumerate(
        top_failures.iterrows(),
        start=1
    ):

        print("\n")
        print("-" * 80)
        print(
            f"FAILURE CASE {number}"
        )
        print("-" * 80)

        print(
            "\nCustomer:"
        )

        print(
            row["customer_text"]
        )

        print(
            "\nPredicted intent:"
        )

        print(
            row["predicted_intent"]
        )

        print(
            "\nConfidence:"
        )

        print(
            row["confidence"]
        )

        print(
            "\nBest historical similarity:"
        )

        print(
            row["best_similarity"]
        )

        print(
            "\nRetrieved examples:"
        )

        print(
            row["retrieved_count"]
        )

        print(
            "\nAction:"
        )

        print(
            row["action"]
        )

        print(
            "\nDecision reason:"
        )

        print(
            row["decision_reason"]
        )

        print(
            "\nHistorical Apple Support reply:"
        )

        print(
            row["historical_brand_reply"]
        )

        print(
            "\nRetrieved reply:"
        )

        print(
            row["retrieved_reply"]
        )

        print(
            "\nFailure score:"
        )

        print(
            round(
                row["failure_score"],
                4
            )
        )

    print("\n")
    print("=" * 80)
    print("FAILURE ANALYSIS SAVED")
    print("=" * 80)

    print(
        "\nFile:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n")
    print(
        "These cases should be manually reviewed"
    )

    print(
        "and used in the final failure-analysis section."
    )


if __name__ == "__main__":
    main()