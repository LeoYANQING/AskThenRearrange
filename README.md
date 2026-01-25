# AskThenRearrange

## 使用教程

下面给出 `task_matter.py` 与 `eval_question.py` 的最小可用示例与参数说明。

### 1) 运行 `task_matter.py`（三种提问策略对比）

默认会依次运行 `direct`、`user_preference`、`parallel` 三种策略，并输出每个场景的：
- Q&A history
- Dialogue summary（基于对话的总结）
- Predicted placements
- Scenario accuracy

```bash
python /home/user/Documents/cdi/AskThenRearrange/task_matter.py
```

常用参数：
```bash
# 指定场景数量
python /home/user/Documents/cdi/AskThenRearrange/task_matter.py --max_scenarios 10

# 指定策略顺序（逗号分隔）
python /home/user/Documents/cdi/AskThenRearrange/task_matter.py --strategies direct,user_preference,parallel

# 指定场景文件路径
python /home/user/Documents/cdi/AskThenRearrange/task_matter.py --scenarios /home/user/Documents/cdi/AskThenRearrange/summarization/scenarios.yml
```

### 2) 运行 `eval_question.py`（多模型评估）

该脚本会评估 **seen / unseen** 两个 split 的预测准确率，并将结果保存到 `result/eval.json`。

```bash
python /home/user/Documents/cdi/AskThenRearrange/eval_question.py --models qwen3,qwen2.5
```

常用参数：
```bash
# 打印每个场景的详细信息（含 prompt/输出）
python /home/user/Documents/cdi/AskThenRearrange/eval_question.py --verbose

# 限制场景数量
python /home/user/Documents/cdi/AskThenRearrange/eval_question.py --max_scenarios 10

# 指定场景文件路径
python /home/user/Documents/cdi/AskThenRearrange/eval_question.py --scenarios /home/user/Documents/cdi/AskThenRearrange/summarization/scenarios.yml
```

结果文件：
- `result/eval.json`（包含每个模型的 seen / unseen 详细信息与平均准确率）

