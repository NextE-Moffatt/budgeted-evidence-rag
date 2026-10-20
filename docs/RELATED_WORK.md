# Related Work and Novelty Audit

Checked against primary sources on 2026-10-20. Venue labels are conservative: an arXiv submission is
not a main-conference acceptance. Summaries are short and provisional pending full-text reproduction.

| Work | Verified status / source | Consequence for this project |
| --- | --- | --- |
| Self-RAG | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html) | Retrieval/critique control through trained special tokens is established. A prompted imitation is not a faithful baseline. |
| Sufficient Context | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) | Relevance and sufficiency differ; separate insufficient evidence from generator failure. |
| Stop-RAG | [arXiv 2510.14337](https://arxiv.org/abs/2510.14337); page lists NeurIPS 2025 MTI-LLM Workshop | A value-based stop/continue MDP trained from full trajectories already exists. This is a closest baseline, not a claimed main-track paper. |
| HALT | [arXiv 2608.02009v3](https://arxiv.org/abs/2608.02009); author page reports Findings of EMNLP 2026 | Claim coverage stopping is established. Generated hop claims are deployable; gold-hop coverage is diagnostic only. Findings is not the main track. |
| When Should Multi-Round RAG Stop? | [arXiv 2608.13237](https://arxiv.org/abs/2608.13237) | Structured sufficiency/gap judges and first-stop evaluation already exist. Count judge overhead and group trajectory prefixes. |
| Adaptive Stopping / MiCP | [arXiv 2604.01413](https://arxiv.org/abs/2604.01413) | Conformal adaptive stopping is already studied; no new guarantee follows from adding a calibration set. |
| ToolChain-CRC | [arXiv 2606.18467](https://arxiv.org/abs/2606.18467) | Trajectory risk control and drift are occupied territory. Check assumptions and implementation before theoretical claims. |
| When Knowledge Is Not Free | [arXiv 2606.02245](https://arxiv.org/abs/2606.02245) | Source prices and evidence budgets are established. Heterogeneous tools alone are not a contribution. |
| AB-RAG | [arXiv 2606.29090](https://arxiv.org/abs/2606.29090) | Training-free budget allocation using confidence/consistency signals already exists. |
| Counterfactual Reasoning for RAG | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1c078897dc08d46091d0d361d9955c6b-Abstract-Conference.html) | Distinguish counterfactual query reasoning/conflict arbitration from estimating continuation benefit and harm. |
| GRIP | [ACL 2026 main, long.196](https://aclanthology.org/2026.acl-long.196/) | Self-triggered retrieval and termination through information planning already exist. |
| R3AG: Retriever Routing | [ACL 2026 main, long.939](https://aclanthology.org/2026.acl-long.939/) | Retrieval quality versus downstream generation utility must be distinguished. |
| SARA | [ACL 2026 main, long.661](https://aclanthology.org/2026.acl-long.661/) | Selective retrieval and compression are not new; compression is outside the initial controlled scope. |
| HippoRAG 2 | [ICML 2025](https://proceedings.mlr.press/v267/gutierrez25a.html) | Graph memory is a different comparison axis; do not add it unless necessary to test transfer. |
| Cost of Agentic Search Decisions | [LREC 2026](https://aclanthology.org/2026.lrec-1.808/) | Budget/latency measurement already has direct work; do not market accounting alone as a new algorithm. |

## Must Read Before the Method Gate

- [ROSE-RAG, author preprint on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7172681):
  source-model ambiguity and robust sequential evidence acquisition may overlap benefit/harm control.
- [Counterfactual Evidence Repair, author preprint on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7549418):
  repair-value supervision under source shift may overlap the proposed transfer angle.

These two full-text audits remain TODO. Do not call their venues verified or assert a settled gap.
Track code availability, exact datasets, cost definitions and official reproduction instructions during
the week-1 audit. No baseline is considered reproduced simply because its paper appears in this table.
