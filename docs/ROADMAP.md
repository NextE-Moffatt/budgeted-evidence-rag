# Roadmap and Decision Log

Start date: 2026-10-20. Indicative research schedule, not a publication or hardware promise.

| Stage | Time | Deliverable / exit gate | Status |
| --- | --- | --- | --- |
| Foundation | Now | Independent repo, literature matrix, protocol, gold-free runtime views, tested smoke | Complete locally; 56 tests passed |
| Novelty / feasibility | Week 1-2 | Full closest-work audit; 200-question training pilot; error taxonomy; split audit and cost profile | Planned |
| Strong baselines | Week 3-4 | Faithful feasible Stop-RAG/HALT/structured controls; matched budgets; generation versus retrieval decomposition | Planned |
| Candidate method | Week 5-6 | Matched rollout labels, lightweight gain/harm model; calibration; no test tuning | Conditional on prior gates |
| Transfer | Week 7-8 | Held-out dataset, second reader, natural and controlled shift; paired uncertainty | Planned |
| Confirmation | Week 9-10 | Freeze configuration, power analysis, preregister comparisons, untouched final evaluation | Planned |
| Writing / audit | Week 11-12 | Claim-to-evidence table, limitations, reproducibility package, target cycle selection | Planned |

## Go / No-Go

1. Novelty: explicitly distinguish the proposal from full-text closest works. If not defensible, pivot.
2. Signal: show sufficient positive/negative continuation cases to support learning and evaluation.
3. Baselines: show improvement over strong controls, not just fixed-k retrieval or a weak confidence proxy.
4. Cost: include controller overhead; stop if savings disappear after accounting.
5. Transfer: do not claim generalization if gains only hold in the training dataset or reader family.

If these fail, retain the framework and report negative findings honestly. Do not scale training simply
to obtain a positive number. Decide whether a benchmark/analysis paper is warranted or select a new topic.

## Immediate Next Experiment

1. Review grouping feasibility on 200 deterministic HotpotQA training records without model calls.
2. Choose an approved reader endpoint and confirm per-run cost limits before real generation.
3. Collect at most 5 x 3 reader calls to validate traces/citations/accounting.
4. Freeze pilot settings and expand to 200 questions only after checking the small run.

No GPU job, background service, model fine-tuning or paid API experiment is running at initialization.

## Decisions

- 2026-10-20: independent general-domain project; GitHub PRIVATE requested by owner.
- 2026-10-20: owner subsequently authorized PUBLIC source publication at
  https://github.com/NextE-Moffatt/budgeted-evidence-rag. Private data and credentials remain excluded.
- 2026-10-20: working name budgeted-evidence-rag; no claim of method novelty.
- 2026-10-20: no inherited data, credentials or service dependencies from prior business projects.
- 2026-10-20: baseline/replay infrastructure first; method training contingent on novelty and signal.
