# Research Proposal

Status: working hypothesis, not a novelty or acceptance claim. Literature cutoff: 2026-10-20.

## Question

Can a stopping policy trained on continuation benefit AND harm transfer across multi-hop QA datasets
more reliably than confidence, coverage, or value-only controls, under a matched total-inference budget?

Working paper theme: **When More Retrieval Hurts: Transferable Stopping in Iterative RAG**.
The eventual title and contribution depend on evidence, not the repository name.

## Why This Is Not Another Agent Wrapper

Stop-RAG already learns continuation value. HALT already verifies expected claims. MiCP already handles
adaptive stopping with conformal prediction. The candidate distinction is not the existence of a
stop action: it is explicit benefit/harm decomposition, cross-dataset transfer, and an audit of whether
apparent savings survive controller overhead and trajectory-level selection.

This distinction may still overlap existing work. Audit Stop-RAG, HALT, S2G-style stopping, ROSE-RAG,
ToolChain-CRC and counterfactual evidence repair before committing to a method paper. If the distinction
does not survive, prefer a rigorous measurement/benchmark contribution or choose another question.

## Falsifiable Hypotheses

1. At the same current answer confidence and evidence coverage, continuation has both beneficial and
   harmful outcomes that are not captured by those signals alone.
2. A lightweight two-output predictor of gain and harm transfers better than a scalar in-domain value
   predictor. Compare on identical features and training trajectories to isolate the decomposition.
3. Gains persist after charging the controller's calls, tokens, and latency, not just retrieval calls.

Candidate label for a fixed next action a and matched reader seed z:

`delta(s,a,z) = F1(answer after a; gold) - F1(answer now; gold)`.

Offline labels distinguish positive, neutral, and negative delta; deployment inputs exclude gold.
Train two probabilities or signed magnitudes, then choose continuation only when estimated net benefit
exceeds its explicit acquisition cost. This is an experimental proposal, not implemented or validated.
Loss choice, features, calibration, and the decision threshold must be selected on non-test splits.

Important: `E[max(delta, 0)] - E[max(-delta, 0)] = E[delta]`. Two heads followed by simple
subtraction are only a decomposition of scalar expected value, not a mathematical novelty. A method
claim needs evidence of better estimation, harm discrimination or transfer under matched capacity and
features. Risk-sensitive decisions may be investigated, but no unproved risk guarantee is implied.
HALT already transfers a HotpotQA-trained verifier across datasets and backbones. Transfer alone is
therefore not novel either. The novelty gate is currently unresolved and must precede method training.

An ordinary full trajectory supplies only one observed continuation. Calling its delta a causal effect
or a cost-optimal action is invalid. Matched branch rollouts, pre-specified alternative queries, repeated
seeds, and uncertainty intervals are needed for counterfactual claims; their collection cost counts.

## Scope

- Start with frozen BM25, frozen reader and fixed query policy; isolate the stopping problem.
- HotpotQA distractor for engineering and feasibility, then MuSiQue for transfer; add 2Wiki only if needed.
- Subsequently test an open-corpus setting and at least a second reader family before broad claims.
- Natural trajectories first. Duplicate, distractor, outdated and contradictory evidence are controlled
  stress tests with separate reporting, not assumed representative real-world frequencies.
- No domain-specific corpus, knowledge-graph service, GUI, full-agent RL or large-model training initially.

## Feasibility Gate

Inspect a deterministic 200-question sample with up to 3 rounds. Measure the frequency and uncertainty
of positive/negative continuation deltas and which errors are retrieval versus reader failures.
If almost all continuations are neutral, harm cannot be reliably labelled, or strong simple baselines
already exhaust the opportunity, change the setting or pivot before expensive training.

Do not infer success from tiny aggregate differences. Use paired question/group-level intervals and
an a priori smallest meaningful improvement (provisional targets: 1 F1 point at matched cost OR 15%
total inference reduction with no more than 1 F1 point loss). These targets are design choices, not
achieved results or statistically justified sample sizes. Estimate confirmatory sample sizes from
pilot variance without reusing pilot questions as the final test.

## Potential Paper Contributions

Only claim contributions supported at the end:

- A controlled trajectory protocol exposing confidence/coverage-matched continuation failure modes.
- A transferable benefit/harm stopping policy outperforming strong matched-budget alternatives.
- Reproducible evaluation of selection effects, controller overhead, and evidence-distribution shift.

A negative result can support a useful study, but does not guarantee a publishable ACL paper.
