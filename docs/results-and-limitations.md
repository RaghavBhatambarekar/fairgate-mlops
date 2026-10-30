# Results, assumptions, and limitations

## Reported results

The project presentation reports these results for a synthetic sample shaped like the Home Credit schema:

| Metric | Original candidate | Mitigated model |
|---|---:|---:|
| Release gate | Blocked | Passed |
| Demographic-parity gap | 0.249 | 0.040 |
| Equal-opportunity gap | 0.414 | 0.022 |
| ROC-AUC | 0.708 | 0.708 |
| Under-30 rejection rate | 29.3% | 14.4% |
| Over-50 rejection rate | 4.5% | 18.4% |
| Overall rejection rate | 14.3% | 17.2% |

The reported mitigation uses Fairlearn `ThresholdOptimizer` with group-specific decision thresholds. The underlying model scores and ROC-AUC are reported as unchanged.

These numbers are transcribed from the project presentation. They have not been independently reproduced from source code in this repository.

## Monitoring demonstration

The presentation reports:
- Fairness alert threshold: rejection-rate gap greater than 0.08.
- Data-drift alert threshold: PSI greater than 0.20.
- At batch 3, the fairness gap is reported as 0.097 while PSI is 0.04.
- A data-drift-only alert would first trigger at batch 6, where PSI is reported as 0.306.

This is a simulated traffic experiment, not evidence from a live lender or real production applicants.

## Dataset scope

The current demonstration results use synthetic data with a Home Credit-like schema. They must not be described as results on the full Home Credit dataset or on real customer records.

## Limitations called out in the presentation

1. **Synthetic sample:** Run the workflow on the full Home Credit dataset (the presentation references 307,511 applicants) before drawing conclusions about that dataset.
2. **Group information at decision time:** `ThresholdOptimizer` uses the applicant's sensitive group at decision time. Some lending policies or jurisdictions may restrict this.
3. **Sensitive attributes:** The presented implementation uses gender and age as sensitive attributes; other relevant sources of disparity may not be represented.
4. **Governance:** The presentation proposes human approval before threshold changes reach production.
5. **Fairness is not one number:** Demographic parity and equal opportunity express different criteria and can conflict. Passing configured thresholds is not proof of universal fairness, legality, or suitability.

## Responsible-use statement

FairGate is an educational prototype. Do not use it to make real credit decisions. Any real-world evaluation would require appropriate legal, privacy, security, model-risk, fairness, and human-governance review.
