# FairGate — viva and code walkthrough guide

This guide explains the project in plain language. It follows the workflow described in the project presentation and the code walkthrough discussed during project preparation. Verify exact function names and implementation details against the original notebook/source before quoting them in a technical defense.

## 60-second project explanation

> FairGate is a fairness-gated MLOps pipeline for credit-default prediction. It trains a LightGBM classifier, tracks experiments with MLflow, and evaluates predictive performance alongside demographic-parity and equal-opportunity gaps. If a candidate fails a fairness check, the workflow attempts post-processing mitigation with Fairlearn's ThresholdOptimizer and evaluates the result again. Only a model that passes the configured gate should be promoted. The project also describes explainable serving through FastAPI and Gradio, plus monitoring for data drift and fairness drift.

## Explain the pipeline like a story

- **Prepare:** clean and validate applicant data, create useful derived features, and split training from test data.
- **Train:** teach LightGBM using labelled examples. It learns patterns associated with default.
- **Gate:** check both predictive performance (ROC-AUC) and group-level fairness metrics.
- **Mitigate:** adjust the post-model decision rule using Fairlearn; this is post-processing, not retraining the LightGBM estimator.
- **Register:** keep only a gate-approved model version as the production candidate.
- **Serve:** accept applicant features through an API or UI and return a risk score and decision explanation.
- **Monitor:** compare later batches with a reference distribution and check for group-level decision gaps.

## Core concepts

### LightGBM
A gradient-boosted decision-tree framework. The classifier estimates a score for the positive class (default). The score is used to rank applicants by estimated risk.

### ROC-AUC
Area under the receiver operating characteristic curve. It measures ranking/discrimination ability, not fairness, calibration, or the quality of a particular operating threshold.

### Demographic parity
Compares the rate of positive decisions across groups. In this project, the presentation describes the corresponding business-facing comparison as group rejection-rate gaps. Be explicit about whether the implementation reports selection or rejection rates and how the positive label is defined.

### Equal opportunity
Compares true-positive rates across groups. Here, the positive class is described as actual default, so the question is whether actual defaulters are identified at similar rates across groups.

### ThresholdOptimizer
A Fairlearn post-processing method that learns group-conditional decision rules under a selected fairness constraint and objective. It does not need to retrain the underlying LightGBM model. It does require sensitive-group information when applying group-specific rules.

### PSI
Population Stability Index compares binned proportions in a reference population and a current population. It is used as a drift signal; it is not a model-accuracy metric.

### TreeSHAP
A method for explaining tree-model predictions by assigning feature contributions. A reason code is a model explanation, not proof of causation or a complete explanation of a lending decision.

### MLflow
Tracks experiment parameters, metrics, artifacts, and model versions/registry metadata.

### FastAPI and Gradio
FastAPI exposes prediction functionality over HTTP. Gradio provides an interactive web interface for users.

## Questions a reviewer may ask

**Why use both performance and fairness checks?**  
A model can rank risk reasonably well and still produce unequal outcomes across groups. The gate treats these as separate release requirements.

**Does excluding sensitive features from the base estimator guarantee fairness?**  
No. Other variables can act as proxies. In addition, the described post-processing stage uses sensitive-group information to choose a decision rule.

**Does mitigation retrain LightGBM?**  
No. ThresholdOptimizer is post-processing: it changes the decision rule applied to model scores rather than fitting a new LightGBM estimator.

**Does passing demographic parity guarantee equal opportunity?**  
No. They are different metrics. The project checks both because satisfying one does not guarantee the other.

**What happens if the mitigated model still fails?**  
The safe release behavior is to block promotion rather than register an unapproved model.

**Does a monitoring alert automatically retrain the model?**  
Only say this if the actual implementation does it. A flag or report that says retraining is required is not the same as an automated retraining workflow.

**Are the reported results from real banking data?**  
No. The presentation states that the reported experiment uses synthetic data with a Home Credit-like schema.

## Presentation discipline

- Distinguish what the code actually executes from what the architecture intends.
- Do not claim a cloud production deployment, automated CI/CD, or automatic retraining unless the corresponding infrastructure and workflow files are present and tested.
- Describe the result values as presentation-reported synthetic-data results until reproduced.
- Explain limitations openly: group data at decision time, limited sensitive attributes, synthetic data, and the need for human governance.
