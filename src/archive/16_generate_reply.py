import os
import pandas as pd
import joblib

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

CLASSIFIER_FILE = "models/intent_classifier.joblib"
VECTORIZER_FILE = "models/intent_reply_vectorizer.joblib"
RETRIEVER_FILE = "models/intent_reply_retriever.joblib"
REPLY_DATABASE_FILE = "models/intent_reply_database.csv"


# ---------------------------------------------------------
# LOAD ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

def load_models():

    print("Loading intent classifier...")

    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    print("Loading reply vectorizer...")

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    print("Loading reply retriever...")

    retriever = joblib.load(
        RETRIEVER_FILE
    )

    print("Loading historical replies...")

    database = pd.read_csv(
        REPLY_DATABASE_FILE,
        low_memory=False
    )

    return (
        classifier,
        vectorizer,
        retriever,
        database
    )


# ---------------------------------------------------------
# RETRIEVE HISTORICAL EXAMPLES
# ---------------------------------------------------------

def retrieve_examples(
    customer_message,
    predicted_intent,
    vectorizer,
    retriever,
    database
):

    query_vector = vectorizer.transform(
        [customer_message]
    )

    distances, indices = retriever.kneighbors(
        query_vector,
        n_neighbors=20
    )

    examples = []

    for distance, database_index in zip(
        distances[0],
        indices[0]
    ):

        row = database.iloc[
            database_index
        ]

        if row["intent"] != predicted_intent:
            continue

        similarity = 1 - distance

        examples.append({

            "customer_text":
                row["customer_text"],

            "brand_reply":
                row["brand_text"],

            "similarity":
                float(similarity)
        })

        if len(examples) == 3:
            break

    return examples


# ---------------------------------------------------------
# LOCAL FALLBACK REPLY
# ---------------------------------------------------------

def create_fallback_reply(
    customer_message,
    intent,
    examples
):

    if len(examples) == 0:

        return (
            "Thanks for reaching out. "
            "We'd like to learn more about the issue "
            "so we can help. Please send us a DM with "
            "more details about your device and what "
            "you're experiencing."
        )

    historical_reply = examples[0]["brand_reply"]

    return (
        "Thanks for reaching out. "
        "Based on similar Apple Support cases, "
        "we'd like to look into this with you. "
        "Please send us a DM with the device model, "
        "iOS version, and details about when the issue "
        "started.\n\n"
        "A similar historical support response was:\n"
        + historical_reply
    )


# ---------------------------------------------------------
# LLM REPLY GENERATION
# ---------------------------------------------------------

def generate_llm_reply(
    customer_message,
    intent,
    examples
):

    if OpenAI is None:

        print(
            "\nOpenAI package is not available."
        )

        return None

    if not OPENAI_API_KEY:

        print(
            "\nOPENAI_API_KEY not found."
        )

        return None

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    evidence_text = ""

    for number, example in enumerate(
        examples,
        start=1
    ):

        evidence_text += (
            f"\nHistorical Example {number}\n"
            f"Customer: {example['customer_text']}\n"
            f"Apple Support Reply: {example['brand_reply']}\n"
            f"Similarity: {example['similarity']:.3f}\n"
        )

    prompt = f"""
You are an AI customer-support drafting assistant
for Apple Support.

Your task is to draft a short reply to the customer.

Customer message:
{customer_message}

Predicted intent:
{intent}

Historical Apple Support examples:
{evidence_text}

Rules:

1. Use the historical examples as evidence.
2. Do not copy a historical reply word-for-word.
3. Do not invent policies, refunds, features, or technical facts.
4. Do not mention the historical examples.
5. Do not mention the intent classification.
6. Do not include another customer's username.
7. If the issue requires investigation, ask the customer
   to continue in DM.
8. Keep the reply concise and professional.
9. Only give advice supported by the historical evidence.
10. If the evidence is weak, ask for more information
    instead of guessing.

Return only the proposed customer-facing reply.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text.strip()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("HIVER AI SUPPORT AGENT")
    print("GROUNDED REPLY GENERATOR")
    print("=" * 80)

    classifier, vectorizer, retriever, database = (
        load_models()
    )

    test_messages = [

        "My iPhone battery is draining very quickly.",

        "Bluetooth is not connecting.",

        "How do I update my iPhone?",

        "iTunes is not working.",

        "My phone keeps freezing after the update."
    ]

    print("\nTesting reply generation...")

    for number, customer_message in enumerate(
        test_messages,
        start=1
    ):

        print("\n")
        print("=" * 80)
        print("TEST CASE", number)
        print("=" * 80)

        print("\nCUSTOMER:")
        print(customer_message)

        # -------------------------------------------------
        # Intent prediction
        # -------------------------------------------------

        predicted_intent = classifier.predict(
            [customer_message]
        )[0]

        probabilities = classifier.predict_proba(
            [customer_message]
        )[0]

        confidence = probabilities.max()

        print("\nPREDICTED INTENT:")
        print(predicted_intent)

        print(
            "CONFIDENCE:",
            round(
                float(confidence),
                4
            )
        )

        # -------------------------------------------------
        # Historical retrieval
        # -------------------------------------------------

        examples = retrieve_examples(
            customer_message,
            predicted_intent,
            vectorizer,
            retriever,
            database
        )

        print("\nRETRIEVED EVIDENCE:")

        for index, example in enumerate(
            examples,
            start=1
        ):

            print(
                f"\nExample {index}"
            )

            print(
                "Similarity:",
                round(
                    example["similarity"],
                    4
                )
            )

            print(
                "Customer:",
                example["customer_text"]
            )

            print(
                "Historical reply:",
                example["brand_reply"]
            )

        # -------------------------------------------------
        # Generate reply
        # -------------------------------------------------

        print("\nGENERATING REPLY...")

        reply = generate_llm_reply(
            customer_message,
            predicted_intent,
            examples
        )

        if reply is None:

            print(
                "\nUsing local fallback."
            )

            reply = create_fallback_reply(
                customer_message,
                predicted_intent,
                examples
            )

        print("\nPROPOSED REPLY:")
        print("-" * 80)
        print(reply)
        print("-" * 80)


if __name__ == "__main__":
    main()