# Health Guide Agent

基于 LangGraph 的**多 Agent 健康管理系统**，已升级为：

- 多 Agent 协作路由（Supervisor + 专家节点）
- 个性化画像持久化记忆（跨会话）
- 本地 RAG 知识增强（可追溯来源）
- 可观测评估（路由、工具使用、时延、引用率）

## 核心能力

### 1) LangGraph 多 Agent

- **Supervisor**：任务分发与结束判断
- **Trainer**：训练计划与 `calculate_tdee`
- **Nutritionist**：饮食策略与营养建议
- **Wellness**：恢复、压力、睡眠管理
- **General**：通用沟通与信息澄清

### 2) 个性化画像（长期记忆）

- 每个用户通过 `user_id` 绑定画像
- 画像存储于 `profile_store.json`
- Agent 可调用：
  - `get_user_profile`
  - `update_user_profile`

### 3) RAG 知识增强

- 本地知识库目录：`knowledge_base/`
- 分层路由：`shared + agent 私有库`
  - `knowledge_base/shared/`
  - `knowledge_base/trainer/`
  - `knowledge_base/nutritionist/`
  - `knowledge_base/wellness/`
  - `knowledge_base/general/`
- 自动读取 `.md / .txt / .pdf` 文档并分块(PDF 通过 `pypdf` 按页提取,支持页级 citation)
- 使用 **Retrieve & Re-rank** 两阶段检索：
  - Stage-1 Dense Retrieve: `BAAI/bge-m3`（多语言,支持中英跨语言检索,8192 长上下文）
  - Stage-2 Cross-Encoder Re-rank: `BAAI/bge-reranker-base`（多语言）
- 工具：`retrieve_health_knowledge`
- 返回内容包含 `source/chunk/score`（并保留 dense/rerank 子分数），便于可追溯

#### 端侧优化（RTX 4060 8GB 友好）

- 模型按需懒加载，减少冷启动内存占用
- GPU 自动启用 FP16 推理（可显著降低显存）
- 检索索引缓存到 `knowledge_base/.index_cache/`，避免重复编码
- 通过环境变量可调 `batch_size / top_k / device`

### 4) 可观测评估

- 每轮记录到 `observability.db`
- 指标：
  - `avg_latency_ms`
  - `retrieval_hit_rate`
  - `citation_rate`
  - 路由分布、工具调用分布
- 会话结束自动导出：`reports/latest_metrics.json`

## 快速开始

1. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

然后编辑 `.env` 并填入你的 API Key：

```ini
SILICONFLOW_API_KEY=your_key
SILICONFLOW_MODEL=Qwen/Qwen2.5-14B-Instruct
METASO_API_KEY=your_key_or_empty
# 可选
PROFILE_STORE_PATH=profile_store.json
KNOWLEDGE_BASE_DIR=knowledge_base

# RAG 可选参数（默认值已针对 8GB 显存 + 中英双语语料调优）
RAG_EMBED_MODEL_NAME=BAAI/bge-m3
RAG_RERANK_MODEL_NAME=BAAI/bge-reranker-base
RAG_DEVICE=auto
RAG_RETRIEVE_TOP_K=12
RAG_FINAL_TOP_K=4
RAG_EMBED_BATCH_SIZE=32
RAG_RERANK_BATCH_SIZE=16
```

2. 使用 Conda 创建环境（推荐）

```bash
conda env create -f environment.yml
conda activate health-guide-rag
```

3. 安装依赖（如已通过 environment.yml 完成可跳过）

```bash
pip install -r requirements.txt
```

4. 下载 RAG 模型（推荐先执行一次）

```bash
python scripts/download_rag_models.py
```

说明：该脚本默认使用 `hf-mirror` 下载（适合中国大陆网络）。

如果你想把模型缓存到指定目录（便于后续离线复用）：

```bash
python scripts/download_rag_models.py --cache-dir .hf_cache
```

如果你希望手动通过环境变量切换下载源：

```bash
# 中国大陆常用
set HF_ENDPOINT=https://hf-mirror.com

# 若需恢复官方源
# set HF_ENDPOINT=https://huggingface.co
```

如果你要禁用脚本内置镜像：

```bash
python scripts/download_rag_models.py --disable-mirror
```

5. 下载知识库语料（权威现成语料）

本项目为 5 个知识库各配置了 1 个推荐权威来源：

- `shared`：WHO《Healthy diet》
- `nutritionist`：USDA/HHS《Dietary Guidelines for Americans》
- `trainer`：WHO《Physical activity》
- `wellness`：WHO《Mental health: strengthening our response》
- `general`：Microsoft Learn《Chit-chat knowledge base》（用于寒暄/日常对话语料）

一键下载到对应目录：

```bash
python scripts/download_knowledge_corpus.py
```

仅下载某一类（示例：trainer）：

```bash
python scripts/download_knowledge_corpus.py --only trainer
```

覆盖已存在文件：

```bash
python scripts/download_knowledge_corpus.py --force
```

下载报告输出到：`reports/knowledge_download_report.json`

说明：`general` 知识库用于处理“你好/谢谢/再见/轻度闲聊”等日常对话，不承载训练或营养专业语料。

6. 启动

```bash
python main.py
```

启动后先输入 `User ID`，即可绑定个人画像并跨会话复用。

## 离线 Embedding 预构建

首次运行前，建议先离线构建索引缓存（减少首轮检索延迟）：

```bash
python scripts/build_rag_index.py --rebuild
```

仅预构建某个 Agent 的私有索引：

```bash
python scripts/build_rag_index.py --agent trainer --rebuild
```

产物：

- `knowledge_base/.index_cache/`（缓存 embeddings/chunks/meta）
- `reports/rag_index_stats.json`（索引统计）

可选参数示例：

```bash
python scripts/build_rag_index.py --chunk-size 420 --overlap 80 --stats-out reports/rag_index_stats.json
```

## RAG 召回准确率评测（Embedding / Rerank 分层评测）

本项目将 RAG 的两个核心阶段 **拆开独立评测**，以便分别判断「embedding 捞得全不全」和
「rerank 排得准不准」：

| 阶段 | 评测对象 | 关注点 | 默认 k |
| --- | --- | --- | --- |
| **Embedding Stage**（Stage-1 Dense Retrieve） | `BAAI/bge-m3` 的稠密召回 | 候选池覆盖能力：相关 chunk 有没有被捞进来 | `5,10,20` |
| **Rerank Stage**（Stage-2 Cross-Encoder Re-rank） | `BAAI/bge-reranker-base` 的头部精排 | 头部精度：最相关的 chunk 是不是被排到了最前面 | `1,3,5` |

评测集默认文件：`eval/rag_eval_dataset.jsonl`

一键运行：

```bash
python scripts/evaluate_rag.py \
  --dataset eval/rag_eval_dataset.jsonl \
  --stage1-ks 5,10,20 \
  --stage2-ks 1,3,5 \
  --stage1-pool 20
```

- `--stage1-pool`：Stage-1 候选池大小，同时也是 Stage-2 重排器输入的候选数
- `--stage1-ks` / `--stage2-ks`：两阶段各自使用的 k 值列表

输出：`reports/rag_eval_report.json`，结构如下：

```text
config                        # 采样数 / k 设置 / 候选池大小
embedding_stage               # Stage-1（仅 embedding）聚合指标
rerank_stage                  # Stage-2（rerank 后）聚合指标
rerank_uplift_vs_embedding    # rerank 相对 embedding 的 Δ（正值=有提升）
per_agent_summary             # 按 agent 拆分的分层指标
details                       # 每条 query 的两阶段 top-N 细节（可追溯）
```

### 指标选择与理由

两个阶段承担的职责不同，因此关注的指标也不同：

1. **Recall@k**（主看 embedding 阶段，k 较大）
   - Embedding 的职责是「把相关片段放进候选池」。只要 Recall@20 足够高，
     下游 rerank 就有翻盘机会；反之 Recall 塌掉，后面再怎么排也救不回来。
   - 这是最能直接反映 embedding 模型质量的指标。

2. **MRR（Mean Reciprocal Rank）**（两阶段都看）
   - 衡量「第一个相关结果的位置倒数」，是单一数字能描述排序好坏的经典 IR 指标。
   - 对 rerank 特别敏感：rerank 的核心价值就是把 Top-1 从错的换成对的。

3. **nDCG@k**（主看 rerank 阶段，k=3/5）
   - 位置感知的排序质量指标：越靠前的相关结果权重越大。
   - LLM 的上下文窗口有限，Top-3/Top-5 的排序质量直接决定 prompt 信息密度，
     nDCG 比 Hit Rate 更能反映这个层面的差异。

4. **MAP@k（Mean Average Precision）**（两阶段都看）
   - 在一条 query 有多个相关文档时（本项目常见），MAP 能同时考虑命中数量和命中位置，
     比 Recall 更细粒度、比 MRR 更能覆盖「全部相关文档」的情形。

5. **Hit Rate@k**（粗粒度健康检查）
   - 「Top-k 中是否至少命中一条相关结果」，作为最直观的冒烟指标保留。

6. **Rerank Uplift = Stage-2 - Stage-1**（隔离 rerank 的边际贡献）
   - 在两阶段共享的 k 上报告 `Δ MRR / Δ Recall@k / Δ nDCG@k / Δ MAP@k`。
   - 健康的 RAG 系统应当出现：Δ Recall@k ≈ 0（rerank 本就不扩大候选池），
     Δ MRR > 0、Δ nDCG@k > 0（rerank 把好结果挤到了前面）。
   - 如果 Δ < 0，说明重排器在这个数据集/语料上反而在伤害结果，需要调参或换模型。

### 评测样本格式（JSONL，每行一个样本）

```json
{"query":"减脂期每天建议热量赤字多少？","agent":"nutritionist","relevant_sources":["nutrition_guidelines.md"]}
{"query":"膝痛用户训练时应注意什么？","agent":"trainer","relevant_sources":["exercise_safety.md","training_recovery.md"]}
```

- `agent`：路由到的 agent 命名空间（`trainer` / `nutritionist` / `wellness` / `general`）
- `relevant_sources`：source 级 ground truth（文件名，自动匹配带命名空间前缀的 source）
- `relevant_chunk_ids`：可选，chunk 级 ground truth（更细粒度）

### A/B 对比多个 embedding 模型

`scripts/compare_embedders.py` 在同一评测集上跑多个 embedding 模型并输出 side-by-side 对比表,用于回答:
「我换模型到底有没有变好?好在哪个指标上?」

```bash
python scripts/compare_embedders.py \
  --models BAAI/bge-small-zh-v1.5,BAAI/bge-m3 \
  --dataset eval/rag_eval_dataset.jsonl
```

说明:

- 每个模型通过**独立子进程**运行(设置 `RAG_EMBED_MODEL_NAME` 环境变量),避免两个
  embedding 模型同时驻留显存,也避免 index cache 串扰(fingerprint 已纳入模型名,
  会自动失效重建)。
- 第一个模型是 baseline,第二个是 candidate;表格最右一列是 Δ(candidate - baseline),
  `↑` 表示 candidate 更好,`↓` 表示更差。
- 适合的对比场景:中文库换多语言库(`bge-small-zh` → `bge-m3`)、从 small 升 base、
  对比 `bge-m3` vs `multilingual-e5-base` 等。

输出:

- `reports/embedder_compare/report_<model>.json` — 每个模型的完整两阶段报告
- `reports/embedder_compare/summary.json` — 合并后的 summary + Δ

### 扩展建议

- 线下迭代 embedding / rerank 模型时，跑一次脚本就能看到每个组件自己的指标曲线，
  避免「端到端只有一个总分、不知道是谁在拖后腿」。
- 建议按 agent 持续补充 10-30 条高质量 query；从面试角度讲，
  能同时展示 **数据集 → 指标 → 分层归因 → 迭代闭环** 的完整评测工程能力。

## RAG 文档维护

将你的知识文档（指南、笔记、FAQ、论文/白皮书 PDF）放到 `knowledge_base/` 下：

- `knowledge_base/shared/`
- `knowledge_base/trainer/`
- `knowledge_base/nutritionist/`
- `knowledge_base/wellness/`
- `knowledge_base/general/`

支持的文件类型：

| 后缀 | 解析方式 | 备注 |
| --- | --- | --- |
| `.md` / `.txt` | 直接按 UTF-8 读取 | 适合 FAQ、内部笔记 |
| `.pdf` | `pypdf` 按页抽取纯文本 | 适合论文、白皮书、官方指南；支持页级 citation |

PDF 处理细节：

- 解析器：`pypdf`（纯 Python，无系统依赖）。通过 `pip install pypdf` 或
  `pip install -r requirements.txt` 即可使用。
- 按页提取：每一页分别 `extract_text()`，清洗多余空行后以 form-feed (`\f`) 作为
  分隔符拼接。
- 分块时保留「chunk → 页码」映射：
  - 单页内的 chunk 的 `page_range = "3"`
  - 跨页 chunk 的 `page_range = "3-4"`
- chunk_id 里会带上页码标记，例如
  `nutritionist/usda_dietary_guidelines.pdf#p5-chunk-12`，便于追溯。
- `retrieve()` / `retrieve_stages()` 返回值会带 `page_range` 字段，LLM 回答时可
  直接用作页级引用。
- 扫描版（纯图片）PDF 没有可抽取文本，解析结果为空会被自动跳过，不会污染索引。
  如需 OCR 支持请自行接入 `pytesseract` / `paddleocr` 等（不在本项目默认范围）。
- 解析失败（加密/损坏）的 PDF 会打印一条 `[RAG][warn]` 日志并跳过，不会阻塞整体
  索引构建。

> 注意：修改/新增/删除任何 PDF 都会改变索引 fingerprint，下次 `build()` 会自动
> 重新 encode。如果想强制重建，可执行：
> `python scripts/build_rag_index.py --rebuild`

建议持续补充高质量文档，让检索覆盖更稳定。

## 项目结构

```
Health-Guide-Agent/
├── main.py
├── requirements.txt
├── knowledge_base/
├── reports/
└── health_guide/
    ├── agents/
    ├── graph.py
    ├── tools.py
    ├── rag.py
    ├── profile_store.py
    ├── observability.py
    ├── config.py
    └── ...
```
