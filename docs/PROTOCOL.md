# Experimental Protocol v0.1

Status: draft, to be frozen before confirmatory experiments. No benchmark scores exist yet.

## Data and Splits

- Maintain train, calibration, validation and untouched confirmatory test partitions.
- All prefixes, alternative branches and repeated seeds for a question stay in its partition.
- Build document-connected groups before hashing groups to splits. Current converter connects ALL
  candidate source documents, not just annotated supporting documents. Report largest component and
  actual split sizes; giant components can make this split unusable. Do not silently fall back.
- Current source-id grouping is exact title-based identity, not complete entity/coreference isolation.
  Audit aliases, duplicates and cross-dataset entities before claiming entity-disjoint transfer.
- Fit on train, calibrate only on calibration, choose architecture/budget trade-offs on validation.
  Freeze model/config/input hashes before first confirmatory run. CLI --confirmatory is only a guard.
- No gold support or answer-based source selection at runtime. Test annotations may be used solely
  for final offline scoring. Keep test results out of iterative prompt/threshold design.

## Controlled Comparisons

Freeze reader version/prompt, retriever/corpus, query policy, context order, decoding and round budget.
Compare fixed 1/2/3 rounds, confidence, sufficiency and answer stability, then faithful Stop-RAG, HALT
and structured stopping baselines. Proxy implementations must be named as proxies. If an official
baseline cannot run, explain it, retain the closest feasible alternative, and narrow the claim.

The current collector runs a full reference trajectory up to the round limit or source exhaustion.
It does not stop at its own confidence. Replay charges the prefix up to the FIRST stopping decision.
This estimates stopping on one fixed path; it cannot evaluate policies changing queries, sources or
earlier actions. Online reruns are required to establish user-visible latency and net inference savings.

## Costs and Failure Handling

- Separate full trajectory collection costs from hypothetical deployed prefix costs.
- Charge every retrieval, reader and controller/judge invocation, including failures.
- Store provider input/output usage; missing usage remains null, never an estimated zero.
- Report serial reader/retrieval latency and controller CPU time separately from end-to-end wall time.
  Index construction is also reported. Replay p50/p95 are not measured online p50/p95.
- Maximum rounds bounds retrieval/reader attempts per question. Output token cap and timeout are
  request settings, not a hard dollar/input-token budget. No monetary claims without verified prices.
- A failed read yields an abstaining failed state. It remains in the denominator. No hidden retries.
  Invalid JSON, out-of-evidence citations, missing citations for an answer and invalid scores fail closed.

## Outcomes

Primary: per-question answer EM/F1, matched-budget Pareto frontier, gain and harm under continuation.
Secondary: gold passage recall (not sentence-level support F1), answered coverage/selective error,
failure rate, retrieval/reader/controller cost, costed latency p50/p95, premature stop and source shift.
Abstentions receive zero answer EM/F1; no-answer data need a separate calibrated task definition.
HotpotQA official sentence support and joint metrics must be added before claiming full official scores.

Aggregate with paired question/group-level bootstrap intervals across independent decoding seeds.
Do not treat correlated prefixes or repeated seeds as independent samples. The current smoke report
has point estimates only; statistical inference and power analysis remain TODO.

## Planned Ablations

Compare signed scalar value to separated gain/harm using identical features/capacity/training data.
Remove calibration, gap features and duplicate/correlation features separately. Compare learned versus
question-generated claims without gold-hop leakage. Evaluate in-domain, dataset transfer and controlled
evidence perturbations separately. Charge training/label-collection costs and amortization explicitly.

## Reproducibility and Release

Each output directory is new and contains input hashes, source fingerprint, effective config, dependency
versions and result artifacts. Collection checkpoints are appended per question to traces.jsonl;
traces.json is finalized on success. Interrupted runs are not silently resumed or counted as complete.
Secrets and raw HTTP errors never enter artifacts. Provider errors are normalized to safe categories.
Raw benchmark data and unpublished artifacts are gitignored. Publication and license require owner review.
