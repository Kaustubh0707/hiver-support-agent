
# Hiver AI Support Agent — Take-Home Assignment

## 1. Problem Statement

This project builds an AI support agent for **AppleSupport** using real historical customer-support conversations from Twitter.

The agent performs three main tasks:

1. Classifies an incoming customer message into a defined support intent.
2. Retrieves historically similar AppleSupport conversations and uses their responses as evidence for drafting support.
3. Decides whether the message should be **auto-handled** or **escalated to a human**.

The system is designed to be conservative: uncertain, sensitive, broad, or device-specific cases are escalated instead of being handled automatically.

---

## 2. Dataset

Primary dataset:

**Customer Support on Twitter**
Kaggle: `thoughtvector/customer-support-on-twitter`

The dataset contains approximately 2.8 million tweets from multiple customer-support brands.

### Dataset processing

| Stage                                 |     Count |
| ------------------------------------- | --------: |
| Tweets loaded                         | 2,811,774 |
| Customer → brand conversation pairs  | 1,016,105 |
| AppleSupport conversation pairs       |    98,576 |
| Unique AppleSupport customer messages |    96,807 |
| Development examples                  |    10,000 |

The dataset was processed by matching inbound customer tweets with their corresponding outbound brand replies.

---

## 3. Selected Brand

The selected brand is:

**AppleSupport**

AppleSupport was selected because it had a large number of conversation pairs, providing sufficient historical examples for intent discovery and retrieval.

---

## 4. Intent Taxonomy

The following 10 intents were created from the AppleSupport data:

| Intent                         | Description                                                       |
| ------------------------------ | ----------------------------------------------------------------- |
| `software_update_issue`      | Problems installing or completing software updates                |
| `battery_issue`              | Battery drain, charging, or battery-life problems                 |
| `performance_freezing`       | Slow performance, freezing, crashing, or restart-related problems |
| `app_or_itunes_issue`        | App Store, iTunes, or application-related issues                  |
| `connectivity_issue`         | Wi-Fi, Bluetooth, cellular, or connection problems                |
| `device_functionality_issue` | Problems with specific device functions or hardware behavior      |
| `account_or_billing_issue`   | Purchases, billing, Apple ID, account, or security-related issues |
| `information_or_how_to`      | Questions asking how to perform a task                            |
| `general_support`            | General requests for help without a clear specific intent         |
| `other`                      | Messages that do not clearly fit another category                 |

---

## 5. System Architecture

```text
Customer Message
       |
       v
Intent Classifier
       |
       +----------------------+
       |                      |
       v                      v
Intent Confidence       Historical Retrieval
                              |
                              v
                     Similar AppleSupport
                       Conversations
                              |
                              v
                     Evidence + Similarity
                              |
                              v
                       Decision Policy
                         /          \
                        /            \
                 AUTO-HANDLE       ESCALATE
```

---

## 6. Intent Classification

### Model

The classifier uses:

* TF-IDF text features
* Logistic Regression
* 10 support intents

The development data contains automatically generated pseudo-labels based on keyword/rule-based labeling.

### Results

| Metric          |           Result |
| --------------- | ---------------: |
| Accuracy        | **79.10%** |
| Macro Precision | **80.03%** |
| Macro Recall    | **79.10%** |
| Macro F1        | **79.27%** |

### Per-intent results

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

---

## 7. Baselines

Two simple baselines were evaluated along with the TF-IDF + Logistic Regression classifier.

| System                       | Accuracy | Macro F1 |
| ---------------------------- | -------: | -------: |
| Majority Class               |    10.0% |    0.018 |
| Keyword Rule Baseline        |    95.6% |    0.954 |
| TF-IDF + Logistic Regression |    79.1% |    0.793 |

### Why is the keyword baseline so strong?

This result is **misleading**.

The development labels were themselves generated using keyword-based rules. Therefore, the keyword baseline is very closely aligned with the labeling process.

The 95.6% result should **not** be interpreted as evidence that the keyword system is better than the classifier on real human-labelled customer messages.

This is one of the main limitations of the current evaluation.

---

## 8. Historical Retrieval

The system retrieves similar historical AppleSupport conversations using:

* TF-IDF vectorization
* Nearest-neighbor similarity search
* Historical customer messages and their corresponding AppleSupport replies

The retrieved conversations provide evidence about how similar issues were handled historically.

The final agent displays the top historical examples and avoids blindly copying a previous response.

---

## 9. Auto-Handle vs Escalation Policy

The final decision policy uses:

* Intent confidence threshold: **0.70**
* Historical similarity threshold: **0.50**

A message can be auto-handled only when the system has sufficient confidence and historical evidence.

The system escalates:

* sensitive account/security issues
* billing-related investigations
* device-specific troubleshooting
* broad or unclear requests
* low-confidence predictions
* cases with insufficient historical similarity
* apparently unresolved issues

This conservative design prioritizes avoiding unsupported automated responses.

---

## 10. End-to-End Evaluation

The final agent was evaluated on **500 AppleSupport conversations**.

| Metric                        | Result |
| ----------------------------- | -----: |
| Examples evaluated            |    500 |
| Auto-handle                   |  14.2% |
| Escalate                      |  85.8% |
| Average intent confidence     | 49.97% |
| Average historical similarity | 58.55% |
| Strong historical evidence    |  44.8% |

### Escalation reasons

| Reason                          | Count |
| ------------------------------- | ----: |
| Intent too broad or unclear     |   163 |
| Low intent confidence           |   138 |
| Device-specific troubleshooting |    67 |
| Low historical similarity       |    40 |
| Sensitive or unresolved issue   |    21 |

The high escalation rate is intentional. The agent is designed to avoid automatically answering cases where confidence, evidence, or safety is insufficient.

---

## 11. Threshold Analysis

Several decision policies were compared:

| Policy            | Confidence | Similarity | Auto-handle Rate |
| ----------------- | ---------: | ---------: | ---------------: |
| Current           |       0.60 |       0.40 |            14.6% |
| Balanced          |       0.70 |       0.50 |             7.6% |
| Conservative      |       0.80 |       0.60 |             5.2% |
| Very Conservative |       0.90 |       0.70 |             2.2% |

The **0.70 / 0.50** policy was selected as the preferred balanced policy because it substantially increases the quality of evidence and confidence among automatically handled cases while remaining usable.

The final evaluation pipeline separately reports the current evaluation policy and threshold experiments.

---

## 12. Failure Analysis

The major observed failure modes are:

### 1. Mixed-intent messages

Example pattern:

> A customer combines an iOS update problem with the phone hanging or freezing.

The classifier may select `performance_freezing` even though the message contains multiple issues.

**Hypothesis:** A single-label classifier is not suitable for messages containing multiple support problems.

**Improvement:** Introduce multi-intent classification or issue extraction.

---

### 2. Broad requests

Example pattern:

> "Can you help me with my iPhone?"

There is not enough information to determine a specific support intent.

**Hypothesis:** The current taxonomy forces broad messages into one of the available classes.

**Improvement:** Add an explicit `unclear_request` intent and ask a clarification question.

---

### 3. Context-dependent issues

Example pattern:

> A customer reports that caller ID and several other functions are not working.

The message requires additional context before selecting a precise intent.

**Hypothesis:** Short support messages often require conversation history rather than isolated-message classification.

**Improvement:** Include previous customer/agent turns as model input.

---

### 4. Device-specific troubleshooting

Example pattern:

> A customer reports a screen or hardware-function problem.

The system can identify the broad issue but cannot safely determine the exact technical cause.

**Hypothesis:** Historical similarity alone is insufficient for device diagnosis.

**Improvement:** Escalate these cases or integrate a verified troubleshooting knowledge base.

---

### 5. Weak historical similarity

Some queries receive a valid retrieved example but have relatively low similarity.

**Hypothesis:** TF-IDF similarity is based mainly on lexical overlap and may fail when customers describe the same problem using different wording.

**Improvement:** Use sentence embeddings or a stronger semantic retrieval model, followed by reranking.

---

## 13. What Is Misleading About My Headline Number?

The most important headline number is the **79.1% intent classification accuracy**.

However, it is misleading if interpreted as real-world customer-support accuracy because the evaluation labels were generated automatically using keyword/rule-based heuristics.

This creates a strong relationship between the labeling process and the keyword baseline.

Similarly, the **95.6% keyword baseline accuracy** should not be treated as a meaningful real-world benchmark.

Therefore, the current numbers should be considered **development evidence**, not human-validated production performance.

---

## 14. Reply Quality Evaluation

A local rule-based reply-quality evaluator was implemented as a development proxy.

It produced an average score of **7/9** across the evaluated examples.

However, all evaluated examples received the same score.

Therefore:

**The 7/9 score is not used as a headline result.**

A proper evaluation requires a human-labelled golden set and/or an LLM-as-judge evaluation with an agreement analysis.

---

## 15. Golden Evaluation Set

A 200-example golden-set structure was created for evaluation.

However, the examples were **not manually labelled**.

Therefore, the project does **not** claim to have completed the assignment's required 150–250 hand-labelled examples.

This limitation is explicitly reported rather than presenting automatically generated labels as human annotations.

---

## 16. Decision Log

The project maintains a decision log containing **15 non-obvious engineering decisions**, including decisions related to:

* brand selection
* intent taxonomy
* pseudo-labeling
* classifier choice
* retrieval approach
* evidence thresholds
* escalation policy
* sensitive issue handling
* conservative automation
* evaluation limitations

File:

```text
outputs/decision_log.csv
```

---

## 17. Project Structure

```text
hiver-support-agent/
│
├── data/
│   ├── raw/
│   │   ├── sample.csv
│   │   └── twcs.csv
│   └── processed/
│       ├── conversation_pairs.csv
│       ├── labeled_conversations.csv
│       ├── selected_brand_data.csv
│       └── training_data.csv
│
├── evaluation/
│   ├── golden_set.csv
│   ├── labeling_guide.txt
│   └── 07_validate_golden_set.py
│
├── models/
│   ├── intent_classifier.joblib
│   ├── intent_reply_vectorizer.joblib
│   ├── intent_reply_retriever.joblib
│   └── intent_reply_database.csv
│
├── outputs/
│   ├── baseline_comparison.csv
│   ├── classifier_confusion_matrix.csv
│   ├── classifier_predictions.csv
│   ├── decision_log.csv
│   ├── decision_threshold_comparison.csv
│   ├── end_to_end_metrics.csv
│   ├── final_evaluation_summary.csv
│   ├── reply_quality_evaluation.csv
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
├── README.md
└── requirements.txt
```

---

## 18. How to Run

### 1. Create environment

```bash
python -m venv .venv
```

### 2. Activate environment

PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

CMD:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python src/03_prepare_data.py
python src/04_select_brand.py
python src/05_discover_intents.py
python src/07_create_training_data.py
python src/08_train_intent_classifier.py
python src/11_improve_retrieval.py
python src/13_evaluate_classifier.py
python src/14_compare_baselines.py
python src/15_evaluate_support_agent.py
python src/18_failure_analysis.py
python src/19_create_decision_log.py
python src/20_evaluate_end_to_end.py
python src/21_tune_decision_thresholds.py
python src/23_final_support_agent_v3.py
python src/24_evaluate_reply_quality.py
python src/25_create_final_evaluation.py
```

The final evaluation summary is written to:

```text
outputs/final_evaluation_summary.csv
```

---

## 19. Reproducibility

The repository contains the scripts, trained models, processed data, evaluation outputs, and decision log required to reproduce the reported development results.

The full raw Twitter dataset is not included in the repository because of its size and dataset distribution considerations.

---

## 20. Limitations

The main limitations are:

1. Intent labels are pseudo-labels rather than human annotations.
2. The keyword baseline is therefore affected by label-generation leakage.
3. The golden set was created but not manually labelled.
4. No human/LLM agreement study was completed.
5. The reply-quality score is only a local rule-based proxy.
6. TF-IDF retrieval can fail on semantically similar but lexically different queries.
7. Single-label classification struggles with mixed-intent messages.
8. The current reply generation is template-based rather than powered by an external LLM.

These limitations prevent the reported metrics from being interpreted as production-level support quality.

---

## 21. What I Would Do With One More Week

### Day 1–2: Human-labelled evaluation

Create a 150–250 example golden set with human labels for:

* intent
* automation decision
* reply quality

Use this as the primary evaluation benchmark.

### Day 3: Improve retrieval

Replace TF-IDF retrieval with sentence embeddings and compare semantic retrieval against the current lexical retriever.

### Day 4: Better reply generation

Use an LLM or stronger local model to generate replies from:

* predicted intent
* customer message
* retrieved historical examples
* explicit escalation policy

### Day 5: LLM-as-judge

Create a rubric covering:

* relevance
* correctness
* historical grounding
* helpfulness
* tone
* unsupported claims

Compare LLM judgments with human judgments.

### Day 6: Error-driven improvements

Focus on:

* mixed intents
* broad requests
* low-similarity retrieval
* device-specific issues
* account/security cases

### Day 7: Final evaluation

Run the complete pipeline on the human-labelled golden set and report:

* intent accuracy / macro F1
* retrieval metrics
* auto-handle precision
* escalation rate
* reply quality
* human/LLM agreement
* failure analysis

---

## 22. Final Takeaway

The project demonstrates an end-to-end support-agent pipeline combining **intent classification, historical-response retrieval, evidence-based decision making, and conservative escalation**.

The current development evaluation achieves **79.1% intent accuracy and 79.27% macro F1**, while the decision policy automatically handles only a minority of cases and escalates uncertain or risky requests.

The most important conclusion is not the raw accuracy number: the evaluation methodology itself needs a human-labelled golden set before the system's real-world support quality can be measured reliably.
