# FairGate architecture

## End-to-end flow

1. **Prepare data** — validate the input schema, engineer features, split the data, and persist prepared datasets.
2. **Train candidate** — fit a LightGBM binary classifier and log parameters, metrics, and artifacts with MLflow.
3. **Evaluate release gate** — evaluate held-out ROC-AUC, demographic-parity gap, and equal-opportunity gap.
4. **Mitigate when needed** — if the candidate fails a fairness check, use Fairlearn `ThresholdOptimizer` as a post-processing step and evaluate the resulting decision rule again.
5. **Register** — register and promote only a model that passes the configured gate.
6. **Serve** — expose predictions through FastAPI and an interactive Gradio application; provide TreeSHAP-based reason codes.
7. **Monitor** — compare production-style batches with a reference population using PSI and group rejection-rate gaps.

## Conceptual component map

| Component | Responsibility |
|---|---|
| Configuration | Central thresholds and training/monitoring parameters |
| Shared utilities | Feature engineering, model-frame conversion, group definitions, metrics |
| Data preparation | Validation, train/test split, processed data persistence |
| Training | LightGBM fit, probability scores, AUC, MLflow run |
| Predictor | Shared scoring and decision logic; reason-code generation |
| Fairness gate | Enforce performance and fairness release criteria |
| Mitigation | Learn post-processing decision thresholds with Fairlearn |
| Registry | Version and promote approved model artifacts |
| Monitoring | Detect population and group-outcome changes |
| API/UI | Serve model predictions and expose reports |

## Important implementation distinction

The presentation describes a pipeline with Docker, DVC, GitHub Actions, FastAPI, and Gradio. The actual source tree and workflow files have not yet been supplied to this repository, so this document records the presented architecture rather than claiming that each component is currently implemented or runnable here.

## Decision semantics

The project uses the positive class for default. A higher model score therefore represents higher estimated default risk. The displayed business decision maps the model's binary decision to `REJECT` or `APPROVE`. Confirm the exact threshold and positive-class conventions in the source code before changing or reusing the implementation.
