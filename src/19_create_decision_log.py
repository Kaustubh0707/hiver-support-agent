import os
import pandas as pd

OUTPUT_FOLDER = "outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "decision_log.csv"
)

DECISIONS = [
    {
        "decision_id": 1,
        "decision": "Select AppleSupport as the target brand",
        "reason": "AppleSupport has a large number of customer-brand conversation pairs, providing enough historical support interactions for training and retrieval.",
        "alternative": "Select another brand such as AmazonHelp or SpotifyCares",
        "tradeoff": "AppleSupport provides strong data coverage, but the taxonomy is specific to Apple-related support."
    },
    {
        "decision_id": 2,
        "decision": "Create customer-to-brand conversation pairs",
        "reason": "The agent needs both the customer's problem and the historical brand response so that previous resolutions can be used as evidence.",
        "alternative": "Use individual tweets independently",
        "tradeoff": "Pair construction is more useful for support modeling but requires processing response_tweet_id relationships."
    },
    {
        "decision_id": 3,
        "decision": "Remove duplicate customer messages",
        "reason": "Repeated customer text can bias both training and retrieval toward duplicated examples.",
        "alternative": "Keep all records",
        "tradeoff": "Removing duplicates reduces available examples slightly but produces a cleaner dataset."
    },
    {
        "decision_id": 4,
        "decision": "Use a 10-intent taxonomy",
        "reason": "The discovered customer issues were grouped into a small set of operationally meaningful support categories.",
        "alternative": "Use a much larger number of fine-grained intents",
        "tradeoff": "Fewer intents simplify classification and escalation, but some different issues are grouped together."
    },
    {
        "decision_id": 5,
        "decision": "Use keyword-based pseudo-labels for development",
        "reason": "Manual labeling was not performed, so keyword rules were used to create development labels and continue building the pipeline.",
        "alternative": "Hand-label 150-250 examples",
        "tradeoff": "This is faster, but pseudo-labels may contain systematic errors and do not satisfy the assignment's hand-labelled evaluation requirement."
    },
    {
        "decision_id": 6,
        "decision": "Use TF-IDF + Logistic Regression for intent classification",
        "reason": "It is lightweight, fast, interpretable, and can run locally without requiring an external API.",
        "alternative": "Use a large language model or transformer classifier",
        "tradeoff": "The local model is cheaper and simpler but has lower semantic understanding for ambiguous messages."
    },
    {
        "decision_id": 7,
        "decision": "Balance the development training set",
        "reason": "The original intent distribution is highly uneven, with general support and update-related messages much more common than some other intents.",
        "alternative": "Train directly on the original distribution",
        "tradeoff": "Balancing improves representation of minority intents but does not reflect their true production frequency."
    },
    {
        "decision_id": 8,
        "decision": "Use intent-aware historical reply retrieval",
        "reason": "Retrieval should prefer historical conversations belonging to the same predicted intent rather than relying only on lexical similarity.",
        "alternative": "Use nearest-neighbor retrieval across all conversations",
        "tradeoff": "Intent filtering improves relevance but can return no evidence when the classifier is wrong."
    },
    {
        "decision_id": 9,
        "decision": "Retrieve 20 candidates before selecting the final evidence",
        "reason": "A larger candidate pool provides more opportunities to find a relevant historical response after intent filtering.",
        "alternative": "Retrieve only the single nearest example",
        "tradeoff": "More candidates improve robustness but increase retrieval computation."
    },
    {
        "decision_id": 10,
        "decision": "Escalate low-confidence classifications",
        "reason": "An incorrect intent can lead to incorrect historical evidence and therefore an unsafe or irrelevant support reply.",
        "alternative": "Always generate a reply",
        "tradeoff": "Escalation reduces automation coverage but improves safety when the model is uncertain."
    },
    {
        "decision_id": 11,
        "decision": "Escalate device-specific troubleshooting cases",
        "reason": "Issues involving hardware behavior, freezing, restarting, or device-specific failures often require investigation rather than a generic automated response.",
        "alternative": "Automatically answer all troubleshooting cases",
        "tradeoff": "This reduces automation but avoids unsupported troubleshooting advice."
    },
    {
        "decision_id": 12,
        "decision": "Escalate account and billing issues",
        "reason": "Billing, account access, payment, password, and security-related issues can require private account information or investigation.",
        "alternative": "Automatically provide a generic resolution",
        "tradeoff": "More human workload is accepted in exchange for avoiding account-specific mistakes."
    },
    {
        "decision_id": 13,
        "decision": "Use historical replies as evidence rather than blindly copying them",
        "reason": "A nearest historical reply can be superficially similar but still inappropriate for the current customer situation.",
        "alternative": "Always return the nearest historical reply",
        "tradeoff": "Evidence-based generation is safer, but the current local fallback still has limited natural-language generation capability."
    },
    {
        "decision_id": 14,
        "decision": "Keep the system runnable without an external LLM API",
        "reason": "The pipeline should run locally without requiring paid API usage or an internet connection for inference.",
        "alternative": "Require an external LLM API",
        "tradeoff": "Local execution improves reproducibility and cost control but limits response-generation quality."
    },
    {
        "decision_id": 15,
        "decision": "Report pseudo-label results separately from human-validated results",
        "reason": "The classifier currently achieves 79.1% accuracy on pseudo-labeled development data, but this number does not represent performance on independently hand-labelled customer examples.",
        "alternative": "Present the 79.1% value as final real-world accuracy",
        "tradeoff": "Being explicit about the limitation makes the evaluation less impressive but more credible."
    }
]


def main():
    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("DECISION LOG")
    print("=" * 80)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df = pd.DataFrame(DECISIONS)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDecision log created successfully.")
    print("\nNumber of decisions:", len(df))

    print("\nDecisions:")
    print("-" * 80)

    for _, row in df.iterrows():
        print(
            f"{row['decision_id']}. "
            f"{row['decision']}"
        )

    print("\nSaved file:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("DECISION LOG COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()