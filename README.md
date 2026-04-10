# AskThenRearrange

AskThenRearrange 是一个面向 household rearrangement 的问答式偏好学习项目：智能体在有限提问预算内，通过与用户多轮对话学习收纳偏好，并预测 seen / unseen objects 的最终放置位置。

当前仓库已经完成一轮可复现的核心 milestone：

- 在 `102` 个 household episodes 上完成四种 questioning strategies 的系统对比
- `User-Preference-First` 在 `budget=5` 时达到 `69.8%` unseen accuracy
- 同预算下显著优于 `Direct Querying` 的 `58.7%`，提升 `+11.1` 个百分点
- 在现有 `Qwen3` 全量实验中，`User-Preference-First` 的 unseen accuracy 峰值为 `71.9%`（`budget=6`）
- `GPT-5-chat` 的 10-episode 子集复现实验保持相同排序，`budget=5` 时 `User-Preference-First = 85.8%`，`Direct Querying = 70.8%`

## 仓库结构

- `test_policy_loop.py`: 结构化策略主实验入口
- `test_raw_llm.py`: Raw LLM baseline，不维护结构化 preference state
- `question_policy.py`: question pattern / strategy 选择逻辑
- `proposers.py`: AO / PE / PI 三类问题生成器
- `state_update.py`: 对回答进行结构化状态更新
- `evaluation.py`: 最终 placement 预测与准确率评估
- `data/scenarios_three_rooms_102.json`: 默认 102-episode 数据集
- `logs/` 与 `plots/`: 已生成实验日志与图表
- `docs/system_overview.md`: 当前实现的技术总览
- `paper_draft_v1/`: 论文草稿与 Study 2 实验材料

## 运行前配置

项目通过环境变量切换 LLM 后端。

```bash
# Ollama / OpenAI-compatible backend 二选一
export LLM_BACKEND=ollama
export LLM_MODEL=qwen3
export LLM_BASE_URL=http://127.0.0.1:11434

# 如果使用 OpenAI-compatible API，还需要：
export LLM_BACKEND=openai
export LLM_MODEL=gpt-5-chat
export LLM_API_KEY=YOUR_KEY
export LLM_BASE_URL=https://your-endpoint/v1
```

`llm_factory.py` 会读取这些变量并为 proposer / oracle / updater / evaluator 统一创建模型实例。

## 快速开始

### 1. 检查数据集

```bash
python data.py --index 0
```

### 2. 跑单个 episode 的单策略实验

```bash
python test_policy_loop.py \
  --mode user_preference_first \
  --budget-list 5 \
  --num-samples 1 \
  --start-index 0
```

可选策略：

- `direct_querying`
- `user_preference_first`
- `parallel_exploration`
- `hybrid_all`

### 3. 跑全量 ablation 并保存图与日志

```bash
python test_policy_loop.py \
  --plot-ablation \
  --num-samples 102 \
  --budget-list 1,2,3,4,5,6,7,8,9,10 \
  --output plots/ablation_full_qwen3.png \
  --ablation-log logs/ablation_full_qwen3.jsonl
```

如果只想比较部分策略，可以加：

```bash
--modes direct_querying,user_preference_first
```

### 4. 跑 Raw LLM baseline

```bash
python test_raw_llm.py \
  --budget-list 0,1,3,5 \
  --num-samples 10 \
  --output-jsonl logs/raw_llm_custom.jsonl \
  --output-plot plots/raw_llm_custom.png
```

其中：

- `budget=0` 对应 no-question baseline
- `test_raw_llm.py` 会先自由提问，再直接基于 `qa_history` 输出最终 placements

## 当前主要结果

基于现有 `logs/ablation_full_qwen3.jsonl`：

| Mode | Budget 5 Seen | Budget 5 Unseen | Best Unseen |
|---|---:|---:|---:|
| Direct Querying | 75.3% | 58.7% | 63.1% @ B=10 |
| User-Preference-First | 79.2% | 69.8% | 71.9% @ B=6 |
| Parallel Exploration | 72.1% | 57.2% | 61.5% @ B=10 |
| Hybrid-All | 77.1% | 68.2% | 69.8% @ B=8 |

这说明当前最稳定的核心发现是：优先询问高层用户偏好规则，比逐个物品直接询问更能泛化到 unseen objects。

## 关键产物

- 图表：[plots/ablation_full_qwen3.png](plots/ablation_full_qwen3.png)
- 主日志：[logs/ablation_full_qwen3.jsonl](logs/ablation_full_qwen3.jsonl)
- 系统说明：[docs/system_overview.md](docs/system_overview.md)
- 论文草稿：[paper_draft_v1/main.tex](paper_draft_v1/main.tex)
- Study 2 指南：[paper_draft_v1/study2_experiment_guide.md](paper_draft_v1/study2_experiment_guide.md)

## 备注

- 根目录旧版 README 中提到的 `task_matter.py` 和 `eval_question.py` 已不再是当前仓库入口。
- 当前仓库尚未提供固定依赖锁文件；如果需要，我可以下一步补一个最小可用的 `requirements.txt` 或 `pyproject.toml`。
