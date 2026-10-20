# Research Integrity

- This is an independent, unpublished, general-purpose RAG research repository.
- Never import private domain data, credentials, service configurations, or results from other projects.
- The owner explicitly authorized PUBLIC source publication on 2026-10-20. This does not authorize
  publication of private data, credentials, experiment artifacts, or an open-source license.
- Do not select an open-source license or upload benchmark data without owner approval and license review.
- Gold answers, supporting facts, split labels, and future trajectory states are evaluation metadata,
  never runtime policy or reader inputs.
- Group all prefixes and document-connected examples before splitting. Report giant components;
  never silently relax the split to improve a result.
- Count unsuccessful requests and abstentions. Record unknown token usage as null, not zero.
- Include controller and data-collection costs. Replay savings are estimates, not measured online savings.
- Synthetic smoke tests are software checks, not benchmark or paper results.
- Do not label a simplified proxy as a faithful reproduction of a published baseline.
- No novelty, statistical guarantee, or performance claim without supporting evidence.
- Manual edits use apply_patch. Run pytest and ruff after code changes.
