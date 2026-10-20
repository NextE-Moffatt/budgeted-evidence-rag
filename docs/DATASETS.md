# Dataset Registry

No external dataset is bundled, downloaded automatically, or uploaded to GitHub.

| Dataset | Initial role | Primary source | License / status |
| --- | --- | --- | --- |
| Synthetic fictional fixtures | Software validation only | src/evidence_rag/fixtures | Original tiny examples; not an empirical benchmark |
| HotpotQA | Pilot and source-domain multi-hop QA | [Official site](https://hotpotqa.github.io/), [official repo](https://github.com/hotpotqa/hotpot) | Official site states CC BY-SA 4.0. Preserve attribution/share-alike terms for any released derivatives. Local converter implemented for labelled distractor records. |
| MuSiQue | Held-out dataset transfer | [Author repository](https://github.com/StonyBrookNLP/musique) | Data terms and included components must be reviewed before download/release. Adapter not implemented. |
| 2WikiMultihopQA | Optional second transfer test | [Author repository](https://github.com/Alab-NII/2wikimultihop) | License/component audit pending. Not required for the first pilot. |

Record original split, URL, retrieval date, file SHA256, version and transformation settings for each
snapshot. Tiny local Hotpot-format unit-test records are fictional, not copied benchmark records.
The pilot uses training-set samples only. A local held-out train subset is not an official leaderboard
test score. Distractor candidates do not imply open-domain retrieval. Do not use benchmark gold titles
to supply retrieval targets. Document grouping uses all candidates and may form giant components.
