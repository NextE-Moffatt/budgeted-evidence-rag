# Dataset Registry

No external dataset is bundled, downloaded automatically, or uploaded to GitHub.

| Dataset | Initial role | Primary source | License / status |
| --- | --- | --- | --- |
| Synthetic fictional fixtures | Software validation only | src/evidence_rag/fixtures | Original tiny examples; not an empirical benchmark |
| HotpotQA | Pilot and source-domain multi-hop QA | [Official site](https://hotpotqa.github.io/), [official repo](https://github.com/hotpotqa/hotpot) | Official site states CC BY-SA 4.0. Preserve attribution/share-alike terms for any released derivatives. Local converter implemented for labelled distractor records. |
| MuSiQue | Held-out dataset transfer | [Author repository](https://github.com/StonyBrookNLP/musique) | Authors specify CC BY 4.0 and warn about seed single-hop dev/test leakage. Check their released exclusion IDs when using those source datasets. Adapter not implemented. |
| 2WikiMultihopQA | Optional second transfer test | [Author repository](https://github.com/Alab-NII/2wikimultihop) | License/component audit pending. Not required for the first pilot. |

Record original split, URL, retrieval date, file SHA256, version and transformation settings for each
snapshot. Tiny local Hotpot-format unit-test records are fictional, not copied benchmark records.
The pilot uses training-set samples only. A local held-out train subset is not an official leaderboard
test score. Distractor candidates do not imply open-domain retrieval. Do not use benchmark gold titles
to supply retrieval targets. Document grouping uses all candidates and may form giant components.

## Pinned Training Snapshot

The explicit `fetch-hotpot --allow-download` command uses
[hotpotqa/hotpot_qa revision 1908d6a](https://huggingface.co/datasets/hotpotqa/hotpot_qa/commit/1908d6afbbead072334abe2965f91bd2709910ab).
It downloads only `distractor/train-00000-of-00002.parquet` and `train-00001-of-00002.parquet`,
verifying their exact sizes and SHA256 values from upstream Git LFS pointers. Files total 331,786,656
bytes. This is the same published training source via its Parquet distribution, not a new benchmark.
The local manifest records source URLs, revision, checksums, license and retrieval date. The adapter
scans all supplied shards using PyArrow batches, samples by seeded ID hash and records scanned counts.
The entire downloaded/converted data remains under gitignored `data/`; do not publish it with source.

## Exploratory Row-API Fallback

If large downloads fail, `fetch-hotpot-pilot --allow-download` obtains at most 200 training rows via
the [official rows API](https://huggingface.co/docs/dataset-viewer/rows). It selects one seeded random
10-row contiguous block per row-index stratum, rejects partial/truncated/misaligned responses, saves
response hashes, and checks the upstream repository revision before and after collection. This is a
stratified block pilot, NOT iid sampling or full-training-set ID-hash sampling. API rows cannot be
cryptographically linked to full-shard checksums by these checks; `final_benchmark_eligible` is false.
The converter carries this provenance into dataset.json. Acquire and verify the complete snapshot
before making confirmatory benchmark claims. No validation or test data are downloaded by this fallback.
