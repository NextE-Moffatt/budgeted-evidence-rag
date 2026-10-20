# Budgeted Evidence RAG

面向 ACL 投稿目标的独立研究项目，研究多轮 RAG 中**何时继续检索、何时停止**。
不依赖任何行业数据、已有业务系统或私有知识库。仓库名为工作名称，不代表已经确立的新方法。

项目仓库：[NextE-Moffatt/budgeted-evidence-rag](https://github.com/NextE-Moffatt/budgeted-evidence-rag)。
这是公开的研究原型，未包含真实数据集、私有配置或实验产物；公开代码不代表已经完成论文实验。

## 研究问题

在给定检索预算下，能否预测“继续检索相对现在回答的收益与伤害”，并将停止策略迁移到
未见过的数据集或证据分布？我们重点区分：缺少证据、已有充分证据但生成失败、继续检索引入干扰。

**创新性尚待验证，当前没有论文实验结论。** Stop-RAG 已研究价值驱动停止，HALT 已研究证据覆盖，
MiCP 已研究自适应停止下的校准。简单增加 Agent 循环、置信度阈值或风险控制并不足以构成贡献。
详见 [研究计划](docs/PROPOSAL.md)、[相关工作](docs/RELATED_WORK.md) 和
[实验协议](docs/PROTOCOL.md)。

## 当前实现

- 基于 `bm25s` 的真实 BM25 检索和显式开启的兼容 Chat Completions 接口的轨迹采集。
- 固定轮数、置信度、充分性和答案稳定性四类简单基线；不是已发表方法的完整复现。
- 全轨迹的首停回放、EM/F1、证据召回、拒答、失败及成本统计。
- HotpotQA 本地格式转换、共享文档连通分组、固定训练/校准/验证/测试划分。
- 固定版本、SHA256 校验的公开训练集下载，以及不调用 LLM 的检索可行性审计。
- 不调用模型的合成轨迹测试，输入与代码指纹、不可覆盖的运行目录。

候选学习式控制器、Stop-RAG/HALT 的忠实复现、MuSiQue 适配和真实基准实验**尚未完成**。
不存在已训练的新模型，也没有已测得的性能提升。

## 快速验证

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/evidence-rag smoke --output runs/smoke-001
```

`smoke` 只使用随包分发的虚构数据，不下载语料、不发送网络请求、不消耗模型额度。
报告中的分数只能检查软件是否按预期工作，不能写进论文结果表。
服务器上已验证的 Python 3.12 依赖版本见 `constraints-py312.txt`；同平台重建时可使用
`pip install -c constraints-py312.txt -e '.[dev]'`。它不是跨平台锁文件；实际依赖版本会进入运行指纹。

## 真实数据 Pilot

先从 [HotpotQA 官方网站](https://hotpotqa.github.io/) 获取训练集；不将原始或转换后数据提交 Git。
第一次仅转换小样本，审查分组统计后再扩大：

```bash
evidence-rag prepare-hotpot --input /path/to/hotpot_train_v1.1.json \
  --source-split train --limit 200 --seed pilot-v1 --output data/hotpot-pilot
```

原始站点不可达时，可使用固定版本的 Hugging Face Parquet 训练快照。下载约 332 MB，
不会下载验证集或测试集；下载命令显式开启网络，校验上游两个分片的大小和 SHA256：

```bash
.venv/bin/python -m pip install -e '.[dev,data]'
evidence-rag fetch-hotpot --allow-download --output data/raw/hotpot-train
evidence-rag prepare-hotpot --input data/raw/hotpot-train \
  --source-split train --limit 200 --seed pilot-v1 --output data/hotpot-pilot
evidence-rag audit-retrieval --dataset data/hotpot-pilot/dataset.json \
  --split train --ks 1 2 3 5 10 --output runs/hotpot-retrieval-audit
```

抽样扫描所有提供的训练分片，按固定问题 ID 哈希取 200 题，不是只取第一个分片。
`audit-retrieval` 只报告原问题单次查询的支持段落召回，不代表回答质量或停止策略效果。

大文件传输失败时，可显式获取官方行接口的探索样本：

```bash
evidence-rag fetch-hotpot-pilot --allow-download --limit 200 --seed pilot-v1 \
  --output data/raw/hotpot-pilot-api
evidence-rag prepare-hotpot --input data/raw/hotpot-pilot-api/rows.json \
  --source-split train --limit 200 --seed pilot-v1 --output data/hotpot-pilot
```

这一路径使用分层连续块采样，保留响应、版本检查及抽样局限，不等同于全训练集 ID 哈希抽样，
也不能用于宣称最终基准成绩。数据来源标记会传入转换后的数据集。

`dataset.json` 保留答案供离线评估；读者和策略只能接收剥离标签的公开视图。
`split_audit.json` 报告连通分量和划分规模。如果共享文档使某些划分为空，先重新设计数据协议，
不能为了跑通实验把同组问题拆开。此转换器是 **distractor 候选集设置**，不是开放全库检索。

在服务器环境中设置 `RAG_LLM_BASE_URL`（以 `/v1` 结尾）、`RAG_LLM_MODEL` 和可选
`RAG_LLM_API_KEY`，不要把密钥写入仓库、命令行参数或对话。连接明文 HTTP 时禁止携带密钥，
环回地址除外。配置好后，显式允许最多 5 题、每题最多 3 次读者调用：

```bash
evidence-rag collect --dataset data/hotpot-pilot/dataset.json \
  --split train --limit 5 --max-rounds 3 --top-k 1 --allow-llm \
  --output runs/pilot-001
evidence-rag evaluate --dataset data/hotpot-pilot/dataset.json \
  --traces runs/pilot-001/traces.json --split train \
  --max-rounds 3 --threshold 0.8 --output runs/pilot-eval-001
```

采集总开销和回放前缀开销分开报告。当前每轮 BM25 + 单次读者回答/自报信号/下轮查询，
不调用外部 Judge；自报充分性不等于经校准的可靠度。最大轮数是调用上限，不是美元或输入 token 硬上限。
测试集采集/评估还必须显式传 `--confirmatory`，该标志仅防误操作，不代替预注册和冻结配置。
扩大采集时可使用显式 `--offset` 分批选择同一划分内按 ID 排序的问题，每批仍最多 50 题。
不要为凑够探索题量而使用保留测试题，也不要把重复采集的问题当作独立样本。

## 推进方式

先完成 200 题可行性分析，再决定是否收集训练轨迹或训练小控制器。候选贡献及否决条件见
[路线图](docs/ROADMAP.md)。不在没有有效信号时直接投入模型后训练，也不承诺 ACL 录用。
当前投稿轮次和截止日期尚未确定。代码使用权和公开许可证待所有者确定，暂不作开源授权。
