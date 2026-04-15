# Study 2 实验 App

这是 Study 2 用户实验的本地 Web 应用原型，用于管理 participant、运行策略对话、收集对话后偏好表，并导出 JSON / CSV 数据。

## 功能

- 实验员创建 participant
- 自动分配 3 个 trial 的策略与场景顺序
- participant 端进行文本对话
- 对话结束后填写偏好表
- experimenter 端查看客观结果
- 本地 JSON 持久化
- CSV 导出
- 单轮失败重试与中断恢复

## 运行环境

建议使用你现有的 conda 环境：

```bash
conda activate behavior
```

如果环境里还没有 Web 依赖：

```bash
pip install -r study2_app/requirements.txt
```

## 启动

在仓库根目录运行：

```bash
/Users/yuanda/miniconda3/envs/behavior/bin/python -m uvicorn study2_app.web:app --host 0.0.0.0 --port 8000 --reload
```

打开：

- 实验员端：`http://127.0.0.1:8000/experimenter`
- 被试端：由实验员端创建 participant 后，打开对应 participant link

## 真实模式模型配置

真实模式依赖仓库当前的 LLM 配置。

默认情况下会读取：

```bash
LLM_BACKEND
LLM_MODEL
LLM_BASE_URL
```

如果你的模型服务在局域网机器上，启动前请先设置正确地址，例如：

```bash
export LLM_BACKEND=ollama
export LLM_MODEL=qwen3
export LLM_BASE_URL=http://你的局域网模型地址:端口
```

## 数据位置

运行时数据会写入：

```text
study2_data/
  participants/
  trials/
  exports/
```

该目录已加入 `.gitignore`。

## 场景配置

V1 使用：

- scene manifest: [study2_app/scene_manifest.json](/Users/yuanda/Documents/AskThenRearrange/study2_app/scene_manifest.json)
- dataset: `data/scenarios_three_rooms_102.json`

如果要切换正式 Study 2 场景，请修改 manifest 中的 `episode_index` 或替换成专用数据集。

## 指标说明

V1 中的 `discussed item` 采用一个简单、可解释的规则：

- 只要某个 object name 在任意一轮的系统问题或 participant 回答中被显式提到，就记为 discussed
- 不把 PE 覆盖推断对象自动算入 discussed

因此导出的指标包括：

- `overall_accuracy`
- `discussed_item_accuracy`
- `undiscussed_item_accuracy`
