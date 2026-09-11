
# Hiver AI Support Agent

An AI-powered customer support agent built for the Hiver SDE Intern take-home assignment.

The system uses historical customer-support conversations from Twitter to:

1. Classify incoming customer messages into support intents.
2. Retrieve historically similar Apple Support conversations and use their responses as evidence.
3. Decide whether a message can be auto-handled or should be escalated to a human.
4. Provide a reason for every escalation decision.
5. Evaluate classification, retrieval, decision policy, baselines, and failure cases.

The project intentionally reports both development performance and evaluation limitations instead of presenting pseudo-label results as production-quality accuracy.

---

## 1. Problem Statement

Customer-support teams receive a large number of repetitive requests. The goal of this project is to build a support agent that can understand an incoming customer message, identify the type of problem, find evidence from previous support conversations, and make a safe decision about whether to answer automatically or involve a human.

For this project, the selected support brand is **AppleSupport**.

The system is intentionally conservative: when intent confidence, historical evidence, or issue safety is insufficient, the system escalates the conversation instead of automatically handling it.

---

## 2. Dataset

The primary dataset is the Kaggle **Customer Support on Twitter** dataset.

It contains approximately 2.8 million tweets from customer-support conversations across many brands.

The complete dataset contains:

* 2,811,774 tweets
* 1,016,105 customer → brand conversation pairs
* 98,576 AppleSupport conversation pairs
* 96,807 unique AppleSupport customer messages

The dataset is not included in this repository because of its size.

Place the downloaded dataset at:

```text
data/raw/twcs.csv
```

The repository contains dataset instructions in:

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

This provides enough historical examples for intent development, retrieval, evaluation, and failure analysis.

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

Historical responses are retrieved as evidence, while the final decision is controlled by intent confidence, historical similarity, and safety/escalation rules.

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

# 6. Training and Classification

## 6.1 Development Labels

The initial development dataset was created using keyword-based pseudo-labeling rules.

The workflow was:

```text
AppleSupport conversations
        |
        v
Keyword-based pseudo-labeling
        |
        v
Balanced development dataset
        |
        v
TF-IDF feature extraction
        |
        v
Classifier training
```

The development dataset contains:

```text
10,000 examples
1,000 examples per intent
```

The original baseline classifier used:

* Word TF-IDF
* Logistic Regression
* 80/20 train-test split

The original Logistic Regression result was:

```text
Accuracy: 79.10%
Macro F1: 79.27%
```

## 6.2 Stronger Classifier

A stronger Word + Character TF-IDF classifier was subsequently evaluated using Linear SVM.

The best tested configuration was:

```text
Word TF-IDF:       n-grams (1, 2)
Character TF-IDF:  char_wb n-grams (3, 5)
Classifier:        LinearSVC
C:                 3.0
```

Development benchmark:

```text
Accuracy: 91.30%
Macro F1: 91.29%
```

This is the current **best pseudo-label development benchmark**.

It must not be interpreted as independently human-validated intent accuracy.

---

# 7. Golden Evaluation Set

The project contains a 200-example golden evaluation set:

```text
evaluation/golden_set_labeled.csv
```

It contains:

* 200 unique AppleSupport customer messages
* Reviewed/provisional intent labels
* Historical brand responses
* Labeling notes
* Auto-handle information where applicable

The labeling guidance is documented in:

```text
evaluation/labeling_guide.txt
```

### Important Qualification

The labels are **provisional/reviewed labels**, not independently double-annotated ground truth.

The set was reviewed during development, but a formal independent second annotator and inter-annotator agreement study were not completed.

Therefore, the evaluation below should be described as a **reviewed/provisional golden-set evaluation**, not as a fully validated human benchmark.

---

# 8. Golden-Set Evaluation

The current Word + Character TF-IDF + LinearSVC classifier was evaluated on the 200 reviewed/provisional examples.

| Metric                |           Result |
| --------------------- | ---------------: |
| Golden-set size       |              200 |
| Accuracy              | **23.00%** |
| Macro F1              | **19.30%** |
| Correct predictions   |               46 |
| Incorrect predictions |              154 |

This result is substantially lower than the 91.30% pseudo-label development benchmark.

That gap is one of the most important findings in the project.

It demonstrates why the pseudo-label development score should not be presented as real-world classification accuracy.

### Per-intent Results

| Intent                         | Precision | Recall |    F1 | Support |
| ------------------------------ | --------: | -----: | ----: | ------: |
| `account_or_billing_issue`   |     0.200 |  0.143 | 0.167 |       7 |
| `app_or_itunes_issue`        |     0.188 |  0.231 | 0.207 |      13 |
| `battery_issue`              |     0.353 |  0.316 | 0.333 |      19 |
| `connectivity_issue`         |     0.143 |  0.167 | 0.154 |       6 |
| `device_functionality_issue` |     0.444 |  0.080 | 0.136 |      50 |
| `general_support`            |     0.131 |  0.421 | 0.200 |      19 |
| `information_or_how_to`      |     0.100 |  0.100 | 0.100 |      10 |
| `other`                      |     0.000 |  0.000 | 0.000 |       7 |
| `performance_freezing`       |     0.308 |  0.200 | 0.242 |      20 |
| `software_update_issue`      |     0.419 |  0.367 | 0.391 |      49 |

---

# 9. Baseline Comparison

The following approaches were compared during development:

1. Majority-class baseline
2. Keyword-rule baseline
3. TF-IDF + Logistic Regression
4. Word + Character TF-IDF + LinearSVC

| Approach                            |        Accuracy |         Macro F1 |
| ----------------------------------- | --------------: | ---------------: |
| Majority Class                      |           10.0% |             1.8% |
| Keyword Rules                       |           95.6% |            95.4% |
| TF-IDF + Logistic Regression        |           79.1% |            79.3% |
| Word + Character TF-IDF + LinearSVC | **91.3%** | **91.29%** |

### Interpretation

The keyword baseline performs better than both trained classifiers on the pseudo-label benchmark.

However, this is not meaningful evidence that keyword rules are better in a real support setting because the development labels themselves were generated using keyword-based rules.

This creates label leakage.

Therefore:

> **The 95.6% keyword-rule result is an artificially favorable development benchmark and should not be presented as real-world performance.**

The stronger 91.30% SVM result is useful for model-development comparison, but it is also only a pseudo-label benchmark.

The reviewed/provisional benchmark is much more revealing:

```text
Accuracy: 23.00%
Macro F1: 19.30%
```

---

# 10. Historical Reply Retrieval

The retrieval system uses:

* TF-IDF vectors
* Cosine similarity
* Nearest-neighbor search
* Intent-aware retrieval
* Historical AppleSupport customer → brand response pairs

The retrieval models/database are stored in:

```text
models/intent_reply_vectorizer.joblib
models/intent_reply_retriever.joblib
models/intent_reply_database.csv
```

For each incoming message, the system retrieves relevant historical conversations and keeps the strongest evidence examples.

Example:

```text
Customer:
"My iPhone battery is draining super fast."

Retrieved historical examples:
- Customer battery-drain problem → Apple Support response
- Customer battery-life problem → Apple Support response
- Customer battery issue after update → Apple Support response
```

Historical responses are used as evidence rather than copied blindly.

---

# 11. Auto-Handle vs Escalate Policy

The decision policy combines:

```text
Intent confidence
        +
Historical similarity
        +
Intent safety
        +
Sensitive/unresolved issue detection
```

The intended policy uses confidence and similarity thresholds around:

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
* High-risk intent categories

The final policy is deliberately conservative.

---

# 12. End-to-End Evaluation

The final support agent was evaluated on:

```text
500 AppleSupport customer messages
```

Measured results:

| Metric                          |          Result |
| ------------------------------- | --------------: |
| Examples evaluated              |             500 |
| Auto-handled                    |              71 |
| Escalated                       |             429 |
| Auto-handle rate                | **14.2%** |
| Escalation rate                 | **85.8%** |
| Average intent confidence       |           0.500 |
| Average historical similarity   |           0.586 |
| Evidence availability           | **97.8%** |
| Strong historical evidence      | **44.8%** |
| Auto-handled average confidence |           0.816 |
| Auto-handled average similarity |           0.742 |
| Weak auto-handled cases         |              33 |
| Weak auto-handled rate          | **46.5%** |

The high escalation rate is intentional: the system prefers escalation when it cannot confidently support a safe automatic response.

The 14.2% auto-handle rate is therefore a **policy outcome**, not a measure of classifier accuracy.

---

# 13. Decision Threshold Analysis

A separate threshold sweep was performed to understand the trade-off between automation and evidence quality.

| Policy      | Auto Rate | Avg Confidence | Avg Similarity |
| ----------- | --------: | -------------: | -------------: |
| 0.60 / 0.40 |     14.6% |          0.815 |          0.741 |
| 0.70 / 0.50 |      7.6% |          0.865 |          0.902 |
| 0.80 / 0.60 |      5.2% |          0.903 |          0.927 |
| 0.90 / 0.70 |      2.2% |          0.957 |          1.000 |

These figures come from the threshold-analysis experiment and are useful for understanding the safety/automation trade-off.

The final end-to-end result should be interpreted using the actual final policy evaluation in Section 12 rather than treating this sweep as the final production metric.

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

The largest escalation drivers are ambiguity and low classifier confidence.

This supports the decision to use escalation as a safety mechanism rather than trying to maximize automation.

---

# 15. Top Failure Modes

## 1. Mixed-Intent Messages

Representative case:

```text
A message combines a software-update problem with the phone hanging or restarting.
```

### Why it fails

The classifier is single-label, so it may select one issue even when the message contains multiple problems.

### Hypothesis

A single-label classifier cannot fully represent multi-intent customer-support messages.

### Improvement

Use multi-label classification or detect multiple issue types before retrieval.

---

## 2. Broad Requests

Representative case:

```text
"Can you help me with my iPhone?"
```

### Why it fails

There is not enough information to identify the customer's actual problem.

### Hypothesis

The message contains insufficient evidence for a specific support intent.

### Improvement

Ask a clarification question instead of forcing a specific intent.

---

## 3. General Support vs Information/How-to Confusion

Representative failure cases include messages where a broad request for support overlaps with a request for instructions.

### Why it fails

The two intents are semantically broad and have overlapping language.

### Hypothesis

The current taxonomy does not create a sufficiently sharp boundary between these categories.

### Improvement

Improve annotation guidelines, add boundary examples, and consider a hierarchical classifier.

---

## 4. Low Historical Similarity

Some messages receive relatively high intent confidence but do not have a sufficiently similar historical support response.

### Why it fails

Intent classification and retrieval quality are separate problems. A confident intent prediction does not guarantee that a close historical resolution exists.

### Improvement

Augment TF-IDF retrieval with sentence embeddings and rerank candidates using both semantic similarity and intent compatibility.

---

## 5. Device-Specific Troubleshooting

Representative cases involve issues where the first customer message lacks information such as:

```text
device model
iOS version
error message
previous troubleshooting steps
```

### Why it fails

A safe resolution often depends on diagnostic context that is not available in the initial message.

### Improvement

Use a structured troubleshooting flow that asks for missing diagnostic information before attempting automatic resolution.

---

# 16. What Is Misleading About My Headline Number?

The most tempting headline number is:

```text
91.30% accuracy
```

This is the best classifier result obtained during development, but it is **not human-validated accuracy**.

It was measured on pseudo-labels generated using keyword-based rules.

The same development setup produces a 95.6% keyword-rule baseline, which strongly suggests that the benchmark is influenced by the label-generation procedure.

When the same SVM is evaluated against the 200 reviewed/provisional examples, performance drops to:

```text
Accuracy: 23.00%
Macro F1: 19.30%
```

Therefore:

> **91.30% should be interpreted as a pseudo-label development benchmark, not evidence of 91.30% real-world intent accuracy.**

The most important lesson is that label quality and benchmark independence matter more than optimizing a model against a convenient pseudo-labeling scheme.

The 200-example reviewed/provisional set is a better direction for evaluation, but it still needs independent annotation and agreement measurement before it can be treated as a production-quality golden benchmark.

---

# 17. Reply Quality Evaluation

A local deterministic reply-quality evaluator was implemented in:

```text
src/24_evaluate_reply_quality.py
```

It evaluated 500 examples.

The current proxy produced:

```text
Average score: 7/9
Median score: 7/9
100% >= 7/9
```

However, these results are **not used as a headline reply-quality metric**.

The evaluator is locally defined and deterministic. It does not provide independent human validation.

The required LLM-as-judge evaluation has **not been executed** in the current environment because no LLM API key is configured.

Human/LLM agreement has therefore also **not been measured**.

A stronger evaluation should use a rubric covering:

* Relevance
* Correctness
* Helpfulness
* Grounding in historical evidence
* Tone
* Safety

and compare LLM-judge scores against human ratings.

---

# 18. Safety and Escalation Philosophy

The agent is intentionally conservative.

It is preferable to escalate an uncertain customer issue than to provide an unsupported troubleshooting response.

Automatic handling therefore requires a combination of:

```text
High intent confidence
        +
Sufficient historical evidence
        +
No sensitive escalation condition
        +
Intent suitable for safe handling
```

This is especially important for:

* Account issues
* Billing/payment issues
* Security-related issues
* Device-specific troubleshooting
* Unresolved issues
* Broad or ambiguous requests

The project treats escalation as a safety mechanism rather than as a failure of the system.

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
│   ├── golden_set_labeled.csv
│   └── labeling_guide.txt
│
├── models/
│
├── outputs/
│   ├── baseline_comparison.csv
│   ├── classifier_metrics.csv
│   ├── decision_log.csv
│   ├── decision_threshold_comparison.csv
│   ├── end_to_end_metrics.csv
│   ├── final_evaluation_summary.csv
│   ├── golden_set_metrics.csv
│   ├── svm_tuning_results.csv
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
│   ├── 18_failure_analysis.py
│   ├── 19_create_decision_log.py
│   ├── 20_evaluate_end_to_end.py
│   ├── 21_tune_decision_thresholds.py
│   ├── 23_final_support_agent_v3.py
│   ├── 24_evaluate_reply_quality.py
│   ├── 25_create_final_evaluation.py
│   ├── 26_test_stronger_classifier.py
│   ├── 27_test_word_char_svm.py
│   ├── 28_tune_svm.py
│   ├── 29_find_suspicious_training_examples.py
│   ├── 30_prepare_golden_set.py
│   └── 31_evaluate_golden_set.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Evaluation Scripts

`30_prepare_golden_set.py` prepares the reviewed/provisional evaluation set.

`31_evaluate_golden_set.py` evaluates the current classifier against that set.

The manual-error-review and LLM-judge scripts are not required for the current reproducible pipeline.

---

# 20. Reproducing the Project

## Step 1 — Clone

```bash
git clone https://github.com/Kaustubh0707/hiver-support-agent.git
cd hiver-support-agent
```

## Step 2 — Create Virtual Environment

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

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — Add Dataset

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

for dataset instructions.

## Step 5 — Prepare Conversations

```bash
python src/03_prepare_data.py
```

## Step 6 — Select AppleSupport

```bash
python src/04_select_brand.py
```

## Step 7 — Discover Intents

```bash
python src/05_discover_intents.py
```

## Step 8 — Create Development Training Data

```bash
python src/07_create_training_data.py
```

## Step 9 — Train Original Classifier

```bash
python src/08_train_intent_classifier.py
```

## Step 10 — Build Retrieval System

```bash
python src/11_improve_retrieval.py
```

## Step 11 — Run Baseline and Classifier Evaluations

```bash
python src/13_evaluate_classifier.py
python src/14_compare_baselines.py
```

## Step 12 — Run Stronger Classifier Experiments

```bash
python src/26_test_stronger_classifier.py
python src/27_test_word_char_svm.py
python src/28_tune_svm.py
```

The best development configuration is Word + Character TF-IDF + LinearSVC with `C=3.0`.

## Step 13 — Prepare and Evaluate Golden Set

```bash
python src/30_prepare_golden_set.py
python src/31_evaluate_golden_set.py
```

This produces:

```text
evaluation/golden_set_labeled.csv
evaluation/golden_set_predictions.csv
outputs/golden_set_metrics.csv
outputs/golden_set_confusion_matrix.png
```

## Step 14 — Run End-to-End Evaluation

```bash
python src/20_evaluate_end_to_end.py
```

## Step 15 — Analyze Decision Thresholds

```bash
python src/21_tune_decision_thresholds.py
```

## Step 16 — Run Failure Analysis

```bash
python src/18_failure_analysis.py
```

## Step 17 — Run Reply-Quality Diagnostic

```bash
python src/24_evaluate_reply_quality.py
```

This is a local proxy evaluation, not an LLM-as-judge evaluation.

## Step 18 — Create Final Evaluation Summary

```bash
python src/25_create_final_evaluation.py
```

The final summary is written to:

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

### 1. Independently Validate the Golden Set

Re-annotate the 200 examples independently and measure inter-annotator agreement.

The goal is to establish a defensible 150–250 example evaluation benchmark rather than relying on provisional labels.

### 2. Rebuild the Classifier Around Human Labels

Train and compare:

* TF-IDF + Logistic Regression
* Word + Character TF-IDF + LinearSVC
* Sentence-embedding classifier
* Small transformer classifier

The independently validated labels should become the primary development target.

### 3. Semantic Retrieval

Replace or augment TF-IDF retrieval with sentence embeddings.

A hybrid retriever can combine:

```text
intent compatibility
        +
lexical similarity
        +
semantic similarity
```

### 4. Better Reply Generation

Use retrieved historical responses as grounded evidence and generate concise, customer-specific responses.

The generator should not invent troubleshooting steps unsupported by the retrieved evidence.

### 5. LLM-as-Judge Evaluation

Run an LLM-based rubric for:

* Relevance
* Correctness
* Helpfulness
* Grounding
* Tone
* Safety

Then compare LLM-judge scores against human ratings and report agreement.

### 6. Multi-Intent Handling

Detect messages containing multiple issues and either:

* handle each issue separately,
* ask for clarification, or
* escalate when safe resolution is uncertain.

### 7. Better Observability

Track:

* Intent confidence
* Retrieval similarity
* Escalation reason
* Reply-quality score
* User outcome
* Human override rate

This would allow the decision policy to be tuned using real support outcomes rather than only offline metrics.

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

The strongest development classifier result is:

```text
91.30% accuracy
91.29% macro F1
```

but it is a **pseudo-label development benchmark**.

The reviewed/provisional evaluation shows:

```text
23.00% accuracy
19.30% macro F1
```

This gap is the most important result for understanding the current system.

The end-to-end agent evaluated 500 messages and:

```text
Auto-handled: 14.2%
Escalated:    85.8%
```

The main product-level finding is that safe automation requires combining:

```text
Intent confidence
        +
Historical evidence
        +
Safety rules
        +
Escalation
```

The current system deliberately favors conservative escalation because an unsupported support response can be more harmful than involving a human.

---

# 24. Limitations

The current version has several important limitations:

* Development labels are pseudo-labels rather than independently validated labels.
* The 200-example evaluation set is reviewed/provisional rather than independently double-annotated.
* The 91.30% classifier result is therefore not a human-validated accuracy claim.
* The keyword baseline is affected by label-generation leakage.
* Human/LLM agreement has not been measured.
* LLM-as-judge evaluation has not been executed in the current environment.
* Reply-quality evaluation is a local deterministic proxy rather than a validated human evaluation.
* Retrieval uses TF-IDF rather than a semantic embedding model.
* The classifier is single-intent.
* The system does not yet use an LLM to generate fully personalized responses.
* Some device-specific issues require diagnostic information that is not present in the first customer message.
* The current evaluation is offline and does not measure real customer outcomes.

These limitations are intentionally documented rather than hidden.

---

# 25. Repository

GitHub:

`https://github.com/Kaustubh0707/hiver-support-agent`

---

# 26. Final Note

This project prioritizes:

* Reproducibility
* Transparent evaluation
* Grounded historical evidence
* Conservative automation
* Safe escalation
* Honest reporting of limitations

The headline classifier number should never be presented without its evaluation context.

The most defensible interpretation of the current results is:

> **The project demonstrates a complete support-agent pipeline and a strong pseudo-label development classifier, but the current evidence does not support a claim of production-level intent accuracy.**
