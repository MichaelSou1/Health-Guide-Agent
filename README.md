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
- 自动读取 `.md/.txt` 文档并分块
- 使用 **Retrieve & Re-rank** 两阶段检索：
  - Stage-1 Dense Retrieve: `BAAI/bge-small-zh-v1.5`
  - Stage-2 Cross-Encoder Re-rank: `BAAI/bge-reranker-base`
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

# RAG 可选参数（默认值已针对 8GB 显存调优）
RAG_EMBED_MODEL_NAME=BAAI/bge-small-zh-v1.5
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

## RAG 评测（Recall@k / MRR / 命中率）

评测集默认文件：`eval/rag_eval_dataset.jsonl`

```bash
python scripts/evaluate_rag.py --dataset eval/rag_eval_dataset.jsonl --ks 1,3,5
```

输出：

- `reports/rag_eval_report.json`
- 包含 `summary`（总体）和 `per_agent_summary`（按 Agent）

默认统计指标：

- `Recall@k`
- `MRR`
- `Hit Rate@k`

评测样本格式（JSONL，每行一个样本）：

```json
{"query":"减脂期每天建议热量赤字多少？","agent":"nutritionist","relevant_sources":["nutrition_guidelines.md"]}
{"query":"膝痛用户训练时应注意什么？","agent":"trainer","relevant_sources":["exercise_safety.md","training_recovery.md"]}
```

## RAG 文档维护

将你的知识文档（指南、笔记、FAQ）放到 `knowledge_base/` 下：

- `knowledge_base/shared/`
- `knowledge_base/trainer/`
- `knowledge_base/nutritionist/`
- `knowledge_base/wellness/`
- `knowledge_base/general/`

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
