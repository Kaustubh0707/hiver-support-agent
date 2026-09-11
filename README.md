
# Hiver AI Support Agent

An AI-powered customer support agent built for the Hiver SDE Intern take-home assignment.

The system uses historical customer-support conversations from Twitter to:

1. Classify incoming customer messages into support intents.
2. Retrieve historically similar Apple Support conversations and use their responses as evidence.
3. Decide whether a message can be auto-handled or should be escalated to a human.
4. Provide a reason for every escalation decision.
5. Evaluate the classifier, retrieval system, decision policy, baselines, and failure cases.

---

## 1. Problem Statement

Customer-support teams receive a large number of repetitive requests. The goal of this project is to build a support agent that can understand an incoming customer message, identify the type of problem, find evidence from previous support conversations, and make a safe decision about whether to answer automatically or involve a human.

For this project, the selected support brand is  **AppleSupport** .

The system is intentionally conservative: when confidence, historical evidence, or issue safety is insufficient, the system escalates the conversation instead of automatically handling it.

---

## 2. Dataset

The primary dataset is the Kaggle **Customer Support on Twitter** dataset.

It contains approximately 2.8 million tweets from customer-support conversations across many brands.

The complete dataset contains:

* 2,811,774 tweets
* 1,016,105 customer → brand conversation pairs
* 98,576 AppleSupport conversation pairs

The dataset is not included in this repository because of its size.

Place the downloaded dataset at:

```text
data/raw/twcs.csv
```

The repository contains instructions in:

```text
data/raw/README.md
```

---

## 3. Why AppleSupport?

AppleSupport was selected because it has a relatively large number of historical support conversations.

After constructing customer → brand response pairs:

```text
AppleSupport conversations: 98,576
```

This provides enough historical examples for intent discovery, retrieval, evaluation, and failure analysis.

---

## 4. System Architecture

```text
Customer Message
       |
       v
Intent Classifier
       |
       v
Predicted Intent + Confidence
       |
       v
Historical Conversation Retriever
       |
       v
Similar Apple Support Examples
       |
       v
Decision Policy
   /           \
AUTO-HANDLE   ESCALATE
   |              |
   v              v
Safe Reply     Human Support
Template       + Reason
```

The agent does not blindly copy historical responses.

Historical responses are retrieved as evidence, while the final decision is controlled by the intent confidence, historical similarity, and safety/escalation policy.

---

## 5. Intent Taxonomy

The project uses ten support intents:

| Intent                         | Description                                                         |
| ------------------------------ | ------------------------------------------------------------------- |
| `software_update_issue`      | Problems installing, downloading, or completing software updates    |
| `battery_issue`              | Battery drain, charging, or battery-life problems                   |
| `performance_freezing`       | Freezing, lagging, restarting, or general performance problems      |
| `app_or_itunes_issue`        | Problems involving applications, App Store, or iTunes               |
| `connectivity_issue`         | Wi-Fi, Bluetooth, network, or connectivity problems                 |
| `device_functionality_issue` | Problems with device hardware or specific device functions          |
| `account_or_billing_issue`   | Apple ID, purchases, billing, payments, or account-related problems |
| `information_or_how_to`      | Questions asking how to perform a task or obtain information        |
| `general_support`            | Broad requests for help without a specific issue                    |
| `other`                      | Messages that do not clearly fit the defined categories             |

---

## 6. Training Approach

The initial development dataset was created using keyword-based pseudo-labeling rules.

The workflow was:

```text
AppleSupport conversations
        |
        v
Keyword-based pseudo-labeling
        |
        v
Balanced training dataset
        |
        v
TF-IDF feature extraction
        |
        v
Logistic Regression classifier
```

The training dataset contains:

```text
10,000 examples
1,000 examples per intent
```

The classifier uses:

* TF-IDF
* Logistic Regression
* 80/20 train-test split

---

## 7. Important Labeling Limitation

The assignment asks for a  **150–250 example hand-labelled golden evaluation set** .

The current implementation does **not** claim to have completed that requirement.

Instead, the development labels were generated using keyword-based rules.

Therefore, the classifier metrics reported below are  **pseudo-label development metrics** , not human-validated accuracy.

This distinction is important because the keyword baseline performs extremely well against labels generated by the same type of rules.

A proper next step would be to create a genuinely human-labelled evaluation set and use it as the primary benchmark.

---

# 8. Classifier Results

The TF-IDF + Logistic Regression classifier achieved:

| Metric          |          Result |
| --------------- | --------------: |
| Accuracy        | **79.1%** |
| Macro Precision | **80.0%** |
| Macro Recall    | **79.1%** |
| Macro F1        | **79.3%** |

Per-intent performance:

| Intent               | Precision | Recall |   F1 |
| -------------------- | --------: | -----: | ---: |
| Account/Billing      |      0.87 |   0.89 | 0.88 |
| App/iTunes           |      0.81 |   0.81 | 0.81 |
| Battery              |      0.99 |   0.88 | 0.93 |
| Connectivity         |      0.91 |   0.86 | 0.88 |
| Device Functionality |      0.81 |   0.84 | 0.82 |
| General Support      |      0.60 |   0.61 | 0.60 |
| Information/How-to   |      0.51 |   0.52 | 0.52 |
| Other                |      0.69 |   0.92 | 0.79 |
| Performance/Freezing |      0.87 |   0.72 | 0.79 |
| Software Update      |      0.94 |   0.86 | 0.90 |

The weakest areas are `information_or_how_to` and `general_support`, which are semantically broad and overlap with other categories.

---

# 9. Baseline Comparison

Three approaches were compared:

1. Majority-class baseline
2. Keyword-rule baseline
3. TF-IDF + Logistic Regression

| Baseline                     | Accuracy | Macro F1 |
| ---------------------------- | -------: | -------: |
| Majority Class               |    10.0% |     1.8% |
| Keyword Rules                |    95.6% |    95.4% |
| TF-IDF + Logistic Regression |    79.1% |    79.3% |

### Interpretation

The keyword baseline performs better than the trained classifier.

However, this result is **not a meaningful proof that keyword rules are better** because the development labels themselves were generated using keyword-based rules.

This creates label leakage.

Therefore, the 95.6% keyword result should not be presented as a real-world benchmark.

A human-labelled test set is required for a fair comparison.

---

# 10. Historical Reply Retrieval

The retrieval system uses:

* TF-IDF vectors
* cosine similarity
* nearest-neighbor search
* historical AppleSupport customer → brand response pairs

For each incoming message, the system retrieves up to three relevant historical conversations.

Example:

```text
Customer:
"My iPhone battery is draining super fast."

Retrieved historical examples:
- Customer battery-drain problem → Apple Support response
- Customer battery-life problem → Apple Support response
- Customer battery issue after update → Apple Support response
```

These historical responses are used as evidence rather than copied blindly.

---

# 11. Auto-Handle vs Escalate Policy

The final policy uses:

```text
Intent confidence >= 0.70
Historical similarity >= 0.50
```

Additional safety rules escalate:

* Account/billing/security-related issues
* Device-specific troubleshooting
* Broad or unclear requests
* Unresolved issues
* Low-confidence predictions
* Low historical similarity

This produces a conservative support policy.

---

# 12. End-to-End Evaluation

The final agent was evaluated on:

```text
500 AppleSupport customer messages
```

Results:

| Metric                        |          Result |
| ----------------------------- | --------------: |
| Examples evaluated            |             500 |
| Auto-handled                  |              71 |
| Escalated                     |             429 |
| Auto-handle rate              | **14.2%** |
| Escalation rate               | **85.8%** |
| Average intent confidence     |           0.500 |
| Average historical similarity |           0.586 |
| Strong historical evidence    | **44.8%** |

For the automatically handled examples:

```text
Average intent confidence: 0.816
Average historical similarity: 0.742
```

The conservative policy intentionally keeps the auto-handle rate relatively low.

---

# 13. Decision Threshold Analysis

Several threshold configurations were tested:

| Policy      | Auto Rate | Avg Confidence | Avg Similarity |
| ----------- | --------: | -------------: | -------------: |
| 0.60 / 0.40 |     14.6% |          0.815 |          0.741 |
| 0.70 / 0.50 |      7.6% |          0.865 |          0.902 |
| 0.80 / 0.60 |      5.2% |          0.903 |          0.927 |
| 0.90 / 0.70 |      2.2% |          0.957 |          1.000 |

The final policy favors safety and evidence quality over maximizing automation.

---

# 14. Main Escalation Reasons

Among the 500 evaluated examples:

| Reason                          | Count |
| ------------------------------- | ----: |
| Broad / unclear intent          |   163 |
| Low classifier confidence       |   138 |
| Device-specific troubleshooting |    67 |
| Low historical similarity       |    40 |
| Sensitive / unresolved issue    |    21 |

This shows that most escalations are caused by ambiguity or insufficient confidence rather than by the retrieval system completely failing.

---

# 15. Top Failure Modes

## 1. Mixed-intent messages

Example:

```text
A message combines a software-update problem with the phone hanging or restarting.
```

The classifier may choose one intent even though multiple issues are present.

### Hypothesis

The current single-label classifier is not designed for multi-intent support messages.

### Improvement

Use multi-label classification or detect multiple issue types before retrieval.

---

## 2. Broad requests

Example:

```text
"Can you help me with my iPhone?"
```

There is not enough information to determine the actual problem.

### Hypothesis

The message does not contain enough lexical evidence for a specific intent.

### Improvement

Ask a clarification question instead of forcing a specific intent.

---

## 3. General Support vs Information/How-to confusion

These two intents frequently overlap.

The largest confusion patterns include:

```text
general_support -> information_or_how_to
information_or_how_to -> general_support
```

### Improvement

Use examples with clearer annotation boundaries and consider a hierarchical classifier.

---

## 4. Low historical similarity

Some messages have a high classifier confidence but no sufficiently similar historical support response.

### Hypothesis

Intent classification and retrieval quality are not perfectly correlated.

### Improvement

Use semantic embeddings in addition to TF-IDF and rerank retrieved responses using both intent compatibility and semantic similarity.

---

## 5. Device-specific troubleshooting

Some issues require information that is not present in the customer's first message.

For example:

```text
device model
iOS version
error message
previous troubleshooting steps
```

### Improvement

Use a structured troubleshooting flow that asks for missing diagnostic information before attempting automatic resolution.

---

# 16. What Is Misleading About My Headline Number?

The most tempting headline number is the classifier's:

```text
79.1% accuracy
```

This number is misleading if presented as real-world support accuracy.

The reason is that the labels used for development and evaluation were generated using keyword-based rules rather than an independently hand-labelled dataset.

The keyword baseline achieves 95.6% on the same pseudo-labels, which strongly suggests that the benchmark is affected by label-generation rules.

Therefore:

> **79.1% should be interpreted as a development result on pseudo-labels, not as evidence of 79.1% real-world intent accuracy.**

A properly hand-labelled 150–250 example golden set should replace this benchmark before making a production-quality claim.

---

# 17. Reply Quality Evaluation

A local rule-based reply-quality evaluator was also implemented.

It evaluated 500 historical examples.

The proxy evaluator produced:

```text
Average score: 7/9
Median score: 7/9
100% >= 7/9
```

However, these results are  **not used as a headline quality metric** .

The evaluator is deterministic and locally defined, and there is currently no independent human/LLM agreement study.

Therefore, the result should be treated only as a development diagnostic.

A stronger evaluation would use:

* Human ratings
* LLM-as-judge with a defined rubric
* Human/LLM agreement measurement
* Pairwise comparison against baseline replies

---

# 18. Safety and Escalation Philosophy

The agent is intentionally conservative.

It is preferable to escalate an uncertain customer issue rather than provide an unsupported troubleshooting response.

Automatic handling is therefore limited to cases where:

```text
High intent confidence
        +
Sufficient historical evidence
        +
No sensitive escalation condition
        +
Intent is suitable for safe handling
```

This is especially important for account, billing, security, and device-specific problems.

---

# 19. Repository Structure

```text
hiver-support-agent/
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│
├── evaluation/
│   ├── golden_set.csv
│   ├── labeling_guide.txt
│   └── 07_validate_golden_set.py
│
├── models/
│
├── outputs/
│   ├── baseline_comparison.csv
│   ├── classifier_confusion_matrix.csv
│   ├── classifier_metrics.csv
│   ├── decision_log.csv
│   ├── decision_threshold_comparison.csv
│   ├── end_to_end_metrics.csv
│   ├── final_evaluation_summary.csv
│   ├── support_agent_evaluation.csv
│   └── top_failure_cases.csv
│
├── src/
│   ├── 03_prepare_data.py
│   ├── 04_select_brand.py
│   ├── 05_discover_intents.py
│   ├── 07_create_training_data.py
│   ├── 08_train_intent_classifier.py
│   ├── 11_improve_retrieval.py
│   ├── 13_evaluate_classifier.py
│   ├── 14_compare_baselines.py
│   ├── 15_evaluate_support_agent.py
│   ├── 18_failure_analysis.py
│   ├── 19_create_decision_log.py
│   ├── 20_evaluate_end_to_end.py
│   ├── 21_tune_decision_thresholds.py
│   ├── 23_final_support_agent_v3.py
│   ├── 24_evaluate_reply_quality.py
│   └── 25_create_final_evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 20. Reproducing the Project

## Step 1 — Clone

```bash
git clone https://github.com/Kaustubh0707/hiver-support-agent.git
cd hiver-support-agent
```

## Step 2 — Create virtual environment

Windows:

```bash
python -m venv .venv
```

PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

CMD:

```bash
.venv\Scripts\activate
```

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — Add the dataset

Download the Customer Support on Twitter dataset and place:

```text
twcs.csv
```

inside:

```text
data/raw/
```

See:

```text
data/raw/README.md
```

for the dataset instructions.

## Step 5 — Prepare conversations

```bash
python src/03_prepare_data.py
```

## Step 6 — Select AppleSupport

```bash
python src/04_select_brand.py
```

## Step 7 — Discover intents

```bash
python src/05_discover_intents.py
```

## Step 8 — Create development training data

```bash
python src/07_create_training_data.py
```

## Step 9 — Train classifier

```bash
python src/08_train_intent_classifier.py
```

## Step 10 — Build retrieval system

```bash
python src/11_improve_retrieval.py
```

## Step 11 — Run evaluations

```bash
python src/13_evaluate_classifier.py
python src/14_compare_baselines.py
python src/20_evaluate_end_to_end.py
python src/21_tune_decision_thresholds.py
python src/18_failure_analysis.py
python src/24_evaluate_reply_quality.py
python src/25_create_final_evaluation.py
```

The final evaluation summary is written to:

```text
outputs/final_evaluation_summary.csv
```

---

# 21. Decision Log

The project contains a 15-entry decision log:

```text
outputs/decision_log.csv
```

The decisions cover areas including:

* Brand selection
* Intent taxonomy
* Training-label strategy
* Class balancing
* Retrieval approach
* Evidence threshold
* Confidence threshold
* Sensitive-issue handling
* Conservative escalation
* Historical response usage
* Evaluation methodology
* Failure analysis
* Reproducibility

Each decision includes the reasoning behind the choice.

---

# 22. One More Week

With one additional week, the highest-priority improvements would be:

### 1. Human-labelled golden set

Create 150–250 independently labelled examples and use them as the primary evaluation benchmark.

### 2. Better intent modelling

Replace the keyword-generated labels with human labels and compare:

* TF-IDF + Logistic Regression
* Sentence embeddings
* Small transformer classifier
* LLM-based classification

### 3. Semantic retrieval

Replace or augment TF-IDF retrieval with sentence embeddings.

### 4. Better reply generation

Use retrieved historical responses as grounded evidence and generate a concise response using an LLM.

### 5. Stronger evaluation

Implement an LLM-as-judge rubric for:

* Relevance
* Correctness
* Helpfulness
* Grounding
* Tone
* Safety

Then measure agreement between human ratings and the LLM judge.

### 6. Multi-intent handling

Detect messages containing multiple issues and either handle each issue separately or escalate when appropriate.

---

# 23. Key Takeaways

The project demonstrates a complete support-agent pipeline:

```text
Historical Support Data
        ↓
Brand Selection
        ↓
Intent Discovery
        ↓
Intent Classification
        ↓
Historical Evidence Retrieval
        ↓
Safety / Confidence Policy
        ↓
Auto-Handle or Escalate
        ↓
Evaluation + Failure Analysis
```

The most important result is not simply the classifier accuracy.

The main finding is that  **safe automation requires combining intent confidence with historical evidence and escalation rules** .

The current system deliberately favors conservative escalation because incorrect support responses can be more harmful than asking a human agent to intervene.

---

## Limitations

The current version has several important limitations:

* Development labels are pseudo-labels rather than independently hand-labelled labels.
* The keyword baseline is therefore affected by label-generation leakage.
* There is no human/LLM agreement measurement yet.
* Reply quality evaluation is a local proxy rather than a validated human evaluation.
* Retrieval uses TF-IDF rather than a semantic embedding model.
* The classifier is single-intent.
* The system does not yet use an LLM to generate fully personalized responses.

These limitations are intentionally documented rather than hidden.

---

## Repository

GitHub:

`https://github.com/Kaustubh0707/hiver-support-agent`

---

## Final Note

This project prioritizes reproducibility, transparent evaluation, and safe escalation over presenting inflated performance numbers.

The reported results should therefore be interpreted together with the evaluation limitations described above.
