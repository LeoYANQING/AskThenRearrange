# CLAUDE.md — AskThenRearrange / PrefQuest

本文件是给 Claude 的项目操作手册。每次对话开始时自动生效。

---

## 项目身份

**PrefQuest** 是一个面向家居整理的偏好学习框架，核心问题是：
> 智能体在有限提问预算内，如何通过提问策略的选择最大化对未见物品的偏好泛化能力？

目标期刊：**IJHCS**（International Journal of Human-Computer Studies，Elsevier）  
当前论文文件：`paper_draft_v1/main.tex`（ACM 模板，待迁移至 Elsevier elsarticle）

---

## 论文当前状态

| 章节 | 状态 |
|---|---|
| Abstract / Introduction / Related Work | ✅ 已按 IJHCS 规范重写 |
| §3 Human–Human Study | ✅ 正文完成，κ 值 / Fisher p 值待填（TODO 标注） |
| §4 System Design (PrefQuest) | ✅ 完成 |
| §5 Simulation Study (Study 1) | ✅ 完成，GPT-5-Chat 全量数据 |
| §6.1–6.6 User Study 方法部分 | ⏳ 设计方案讨论中 |
| §6.7–6.9 User Study 结果部分 | ⏳ 待 Study 2 数据收集后填写 |
| §7 Discussion §5.4 / Conclusion | ⏳ 待 Study 2 后补充 |
| Appendix D–F（场景材料/拉丁方/访谈提纲）| ⏳ 待 Study 2 设计定稿后填写 |

---

## 写作任务规范（必须执行）

**每次润色、扩写、新写 main.tex 任何部分之前**，必须按顺序读取：

1. `docs/writing_style_guide.md` — IJHCS 格式、统计、术语规范
2. `docs/writing_logic_guide.md` — 论证结构与叙事逻辑规范

不读这两个文件直接写作会产生格式错误和论证漏洞。

---

## 术语红线（不查文件也必须记住）

| 概念 | 正确写法 | 禁止写法 |
|---|---|---|
| Parallel Exploration 策略 | **PAR** | ~~PE~~（PE 已被 Preference-Eliciting 占用） |
| Hybrid-All 策略 | **HYB** | ~~HA~~ |
| Preference-Eliciting 问题元素 | **PE** | ~~PE strategy~~ |
| Preference-Induction 问题元素 | **PI** | ~~PS~~（PS 是 HHI 编码中的旧术语） |
| 主指标 | **unseen PSR**（Preference Satisfaction Rate） | ~~unseen accuracy~~（已统一为 PSR） |
| 预算 | **$B$**（数学模式） | ~~budget B~~（非数学模式） |
| p 值格式 | **p = .001**（无前导零） | ~~p = 0.001~~ |

---

## 关键实验结论（不得引用错误数字）

以下数字来自 GPT-5-Chat 全量实验（n = 102 episodes），是论文 Abstract 和 Results 的权威数字：

- UPF unseen PSR @ B=5：**85.2%** ± 1.6 SE
- DQ unseen PSR @ B=5：**72.7%** ± 2.1 SE
- 差值：**+12.5 pp**，Wilcoxon W = 94，p < .001
- HYB vs UPF @ B=5：p = .085（不显著）
- HYB vs UPF @ B=10：p = .796（不显著）

Qwen3-8B 的数字（UPF 69.8% / DQ 58.7%）仅用于 Appendix C，不得出现在主文。

---

## 关键文件地图

```
AskThenRearrange/
├── CLAUDE.md                          ← 本文件
├── README.md                          ← 项目说明（人类读者）
├── paper_draft_v1/
│   └── main.tex                       ← 论文主文件（唯一权威版本）
├── docs/
│   ├── writing_style_guide.md         ← IJHCS 格式/统计/术语规范
│   ├── writing_logic_guide.md         ← 论证结构与叙事逻辑规范
│   ├── system_overview.md             ← PrefQuest 技术架构说明
│   └── study2_frontend_PRD.md         ← Study 2 前端系统需求文档
├── logs/
│   └── ablation_full_qwen3.jsonl      ← Qwen3 全量实验日志
└── plots/
    └── ablation_full_qwen3.png        ← 消融实验图
```

---

## 代码任务规范

**代码任务前**先读 `docs/system_overview.md` 了解模块结构。

**运行环境**：conda `behavior`

```bash
conda activate behavior

# LLM 后端配置（二选一）
export LLM_BACKEND=openai
export LLM_MODEL=gpt-5-chat
export LLM_API_KEY=YOUR_KEY
export LLM_BASE_URL=https://your-endpoint/v1

# 或 Ollama
export LLM_BACKEND=ollama
export LLM_MODEL=qwen3
export LLM_BASE_URL=http://127.0.0.1:11434
```

**主要入口**：
- 策略实验：`python test_policy_loop.py`
- Raw LLM baseline：`python test_raw_llm.py`
- Study 2 前端：`study2_app/backend/main.py`
