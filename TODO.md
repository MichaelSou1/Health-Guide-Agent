# Refactor TODO: Agent 架构深度优化

## P0 — 核心架构重构

### 1. 父子 Agent 层级 + 上下文隔离
**问题**：每个专家节点拿到完整 `state["messages"]`，包含其他专家的工具调用 trace 和全量历史，造成上下文噪声与 token 浪费。

**目标**：主 agent（Dispatcher）作为父 agent，调用专家时构造隔离的独立上下文，而非传入整个历史。

**改造点**：
- `dispatcher.py`：不再直接路由到专家节点，改为通过 **tool calling** 方式调用专家（每个专家包装成一个 tool）
- 父 agent 构造子 agent 输入，只包含：
  ```
  SystemMessage(用户画像裁剪片段 + 同伴 scratchpad + RAG 结果)
  HumanMessage(contextualized_query)
  ```
- 专家函数作为 tool 的实现体，返回结构化结果（回答文本 + scratchpad 摘要 + 引用来源）
- `graph.py`：移除 Trainer/Nutritionist/Wellness/General 作为独立 LangGraph 节点，改由 Dispatcher tool call 驱动

**关键约束**：专家之间无直接消息传递，peer scratchpad 由父 agent 在调用下一个专家时注入。

---

### 2. RAG 按需工具化（而非强制 pre-fetch）
**问题**：每个专家节点执行前无条件调用 `retrieve_*_knowledge()`，即便用户问的是问候语或纯个人信息类问题，也会白跑 embedding + rerank。

**目标**：RAG 变为专家可选调用的 tool，由专家自行判断是否需要检索。

**改造点**：
- 移除 `dispatcher.py` 中的 pre-fetch 逻辑（当前在专家 node 函数顶部调用 retrieve）
- 各专家的 tool list 保留 `retrieve_*_knowledge`，让 ReAct loop 自行决策是否调用
- 相应更新各专家 system prompt，说明"如需知识库支持可主动调用工具"

---

## P1 — 性能优化

### 3. ReplanJudge 触发条件收紧
**问题**：当前每次专家执行完都经过 ReplanJudge，即便 `plan` 里还有待执行的专家，也会白调一次 LLM。

**改造点**：
- `graph.py`：在路由函数 `_route_after_dispatch()` 中加条件：仅当 `plan == []` 时才路由到 ReplanJudge
- `plan` 非空时直接继续执行下一个专家

**预期收益**：减少 30–40% 的 LLM 调用次数（多专家计划场景）。

---

### 4. 专家并行 fan-out
**问题**：Planner 输出 `["Trainer", "Nutritionist"]` 时两者强制串行，Nutritionist 等 Trainer 完成才开始，存在明显延迟浪费。

**目标**：无数据依赖的专家并行执行，后处理阶段再合并 scratchpad。

**改造点**：
- `graph.py`：改造路由支持 parallel fan-out（LangGraph Send API）
- Aggregator 等待所有并行专家完成后统一聚合
- peer scratchpad 改为"事后注入"：并行场景下专家无法读取彼此 scratchpad，Aggregator 做最终整合时补偿

**注意**：此改动与改造 1（tool calling 方式调专家）存在架构选择，需确认最终方案后实施。

---

## P2 — 质量提升

### 5. Profile 字段按专家裁剪
**问题**：`profile_to_prompt_text()` 把完整 JSON profile 注入每个专家的 system prompt，各专家拿到大量无关字段（Trainer 不需要饮食偏好，Nutritionist 不需要训练周期细节）。

**改造点**：
- `profile_store.py` 或各专家文件：为每个专家定义 profile 字段白名单
  ```python
  TRAINER_PROFILE_FIELDS = ["age", "weight_kg", "height_cm", "fitness_level", "injuries", "workout_history"]
  NUTRITIONIST_PROFILE_FIELDS = ["age", "weight_kg", "height_cm", "dietary_restrictions", "health_goals", "allergies"]
  ```
- 构建专家 system prompt 时只注入白名单字段

---

### 6. Critic 接入 Safety KB
**问题**：Critic 的安全审查完全依赖 LLM 训练数据，系统提示里只有几条硬编码规则，在特定药物禁忌、特定疾病运动限制等细粒度场景下存在遗漏风险。

**改造点**：
- `tools.py`：新增 `retrieve_safety_guidelines(query)` tool，指向专用安全知识库
- `critic.py`：在 Critic 系统提示中说明可调用安全知识库 tool，并在 draft_answer 审查前先检索相关安全条目
- `knowledge_base/`：新建 `safety/` 子目录，整理运动禁忌、医疗边界等文档

---

## 实施顺序

```
改造 1（父子 Agent + tool calling）
    ↓
改造 2（RAG 按需）          ← 与改造 1 同步进行，均涉及专家调用方式
    ↓
改造 3（ReplanJudge 收紧）  ← 独立，改一行路由条件
    ↓
改造 4（并行 fan-out）      ← 依赖改造 1 架构稳定后再动
    ↓
改造 5 + 6                  ← 独立，随时可做
```
