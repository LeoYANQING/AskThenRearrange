# Study 2 Frontend — PrefQuest

Vite + React + TypeScript 单页应用。双面板布局：左侧实验员控制台，右侧参与者视图。后端契约见 `study2_app/backend/`。

## 目录

- [运行](#运行)
- [布局](#布局)
- [文件地图](#文件地图)
- [会话状态机](#会话状态机)
- [后端 API 契约](#后端-api-契约)
- [中英文显示层（name_mapping）](#中英文显示层name_mapping)
- [语音 STT 流程](#语音-stt-流程)
- [AgentState 监控面板](#agentstate-监控面板)
- [常见开发任务](#常见开发任务)

---

## 运行

```bash
# 首次安装（从仓库根目录）
bash bootstrap.sh

# 开发模式
cd study2_app/frontend
npm run dev            # http://localhost:5173

# 或者从仓库根同时启动前后端
bash study2_app/start.sh
```

Vite dev server 默认代理到 `http://localhost:8001`（见 `vite.config.ts`）。后端端口不同时通过 `PREFQUEST_BACKEND_URL` 覆盖：

```bash
PREFQUEST_BACKEND_URL=http://localhost:8000 npm run dev
```

生产构建：

```bash
npm run build          # → dist/
npm run preview        # 本地预览构建产物
```

---

## 布局

```
┌─ panel-left ────────────┐  ┌─ panel-right ─────────────────┐
│ ExperimenterPanel       │  │ ParticipantView              │
│  - 会话元信息            │  │  (按 phase 切换子视图)         │
│  - 拉丁方行 / 试验顺序   │  │                                │
│  - 载入场景 / 开始对话   │  │  SceneIntro                   │
│  - 结束对话 / 计算分数   │  │  DialogueView                 │
│  - AgentStatePanel      │  │  PreferenceForm               │
│    (仅实验员可见)       │  │  PredictionView               │
│                         │  │  FinalRanking / SessionReport │
└─────────────────────────┘  └───────────────────────────────┘
```

- **实验员面板**：控制流程推进，监控 AgentState。
- **参与者视图**：只展示当前阶段需要的内容。每个 phase 对应一个子组件。

---

## 文件地图

```
src/
├── App.tsx               ← 根组件 + SessionContext（全局会话状态）
├── api.ts                ← 所有后端调用（axios），每个 HTTP 端点一个函数
├── types.ts              ← 所有 TS 类型（AgentState / Trial / Session / 标签映射）
├── voice.ts              ← 浏览器录音 + 上传到 /voice/stt 的封装
├── styles.css            ← 全局样式（暗色主题）
├── main.tsx              ← Vite 入口
└── components/
    ├── SessionCreate.tsx    输入参与者 ID / 拉丁方行 / 预算 → 建会话
    ├── ExperimenterPanel.tsx 左侧总控
    ├── SceneIntro.tsx       场景介绍（房间 / receptacles / 物品清单）
    ├── DialogueView.tsx     多轮问答主体，含语音输入和"需要放置的物体"提示
    ├── PreferenceForm.tsx   对话结束后参与者手动分配物品到 receptacle
    ├── PredictionView.tsx   展示模型预测 vs 参与者金标，含 PSR
    ├── FinalRanking.tsx     3 轮结束后对 3 种策略排序 + 留言
    ├── SessionReport.tsx    实验完成后的完整报告
    └── AgentStatePanel.tsx  仅实验员可见的 AgentState 快照（5 个字段块）
```

---

## 会话状态机

`SessionSnapshot.phase` 驱动参与者视图切换：

```
created           ← 会话刚建/单轮刚结束，等待"载入场景"
  ↓ startTrial
scene_intro       ← 展示场景，等"开始对话"
  ↓ startDialogue
dialogue          ← 多轮问答，参与者答题；实验员可"结束对话"
  ↓ stopDialogue 或 预算用尽
dialogue_complete ← 对话结束，等参与者填偏好表
  ↓ submitPreferenceForm
preference_form   ← 参与者已提交，等实验员"开始评分预测"
  ↓ computeScore
(回到 created 进入下一轮，或 final_ranking)
  ↓ submitFinalRanking
completed
```

一个会话走 3 轮试验（DQ / UPF / PAR，拉丁方顺序），共享同一 `session_id`。`current_trial_index` 指向当前轮。

---

## 后端 API 契约

所有端点在 [src/api.ts](src/api.ts) 中集中管理，返回值类型在 [src/types.ts](src/types.ts)。

| 方法 | 端点 | 用途 |
|---|---|---|
| POST | `/sessions` | 新建会话 |
| GET | `/sessions/{id}` | 拉取会话快照（刷新状态） |
| POST | `/sessions/{id}/trial` | 载入一轮场景（room_type + episode_index） |
| POST | `/dialogue/{id}/start` | 开始对话，返回第一个问题 |
| POST | `/dialogue/{id}/answer` | 提交参与者回答，返回下一个问题或 `dialogue_complete=true` |
| POST | `/dialogue/{id}/stop` | 实验员强制结束对话 |
| POST | `/sessions/{id}/preference_form` | 参与者提交物品→receptacle 分配 |
| POST | `/sessions/{id}/score` | 后端评分，返回 seen/unseen/total PSR |
| POST | `/sessions/{id}/final_ranking` | 3 轮后提交策略排名和主观反馈 |
| POST | `/voice/stt` | 录音 WAV → Dashscope paraformer 转写文本 |

Vite 代理（见 `vite.config.ts`）转发的前缀：`/sessions`、`/dialogue`、`/logs`、`/health`、`/voice`。

---

## 中英文显示层（name_mapping）

- **后端**：所有内部状态（物品名、receptacle、策略逻辑）使用**英文**。
- **前端**：通过 `TrialSnapshot.name_mapping: Record<en, zh>` 渲染中文标签给参与者看。
- 渲染规则：优先显示 `name_mapping[en]`；没有映射则回退到英文原文。

凡是给参与者看的字符串（物品名、receptacle 名、对话内容），都走 `name_mapping`。实验员面板的 AgentState 可以同时显示中英（调试用）。

对话里的问题 `pending.question_zh` 是后端直接翻译好的中文版本，前端无需再处理。

---

## 语音 STT 流程

`voice.ts` 封装了浏览器录音 → WAV → 后端 Dashscope 的完整链路：

1. `Recorder.start()` 用 `getUserMedia` + `AudioContext` 拿到 PCM。
2. `Recorder.stop()` 把 Float32Array 编码成 16-bit PCM WAV（`encodeWAV`）。
3. `transcribe(blob)` POST 到 `/voice/stt`，后端调 Dashscope `paraformer-realtime-v2` 返回文本。

后端要求 `DASHSCOPE_API_KEY` 环境变量。未设置时 `/voice/stt` 返回 500，前端在 `DialogueView` 显示错误。

DialogueView 里的使用模式：按住说话 → 松开自动 stop + transcribe → 填入答题框（参与者可以再编辑后提交）。

---

## AgentState 监控面板

`AgentStatePanel.tsx` 给实验员看的诊断视图，展示当前 trial 的 `AgentStateSnapshot`：

| 字段 | 含义 |
|---|---|
| `confirmed_actions` | 已确认的 object→receptacle 放置（问答中直接问出来的） |
| `negative_actions` | 已排除的 object→receptacle 对（参与者说 no） |
| `confirmed_preferences` | 已归纳的偏好（LearnedPreference：hypothesis + covered_objects + receptacle） |
| `negative_preferences` | 被参与者否决的偏好假设 |
| `unresolved_objects` | 尚未解决（既没确认也没覆盖）的物品 |

面板在每次 QA 后自动刷新（通过 `getSession` 拉最新快照）。方便实验员观察策略内部状态、发现异常行为。

DialogueView 在每轮问答下方会用 `newActionsForTurn` 对比前后 `state_after` 差异，弹出"🔔 需要放置的物体"的 hint —— 提示参与者接下来应该物理操作哪些物品。

---

## 常见开发任务

**改端点**：只需同时改 `api.ts` 里的函数和 `types.ts` 里的响应类型；组件一般不用碰。

**加新的 phase**：
1. 在 `types.ts` 的 `PHASE_LABELS` 加中文标签。
2. 在 `ParticipantView.tsx` 加一个对应的子组件分支。
3. 在 `ExperimenterPanel.tsx` 加对应的推进按钮。

**调整 AgentState 展示**：改 `AgentStatePanel.tsx` 和 `types.ts` 里的 `AgentStateSnapshot`。后端来源在 `backend/session_store.py::agent_state_snapshot`。

**样式**：`styles.css` 单文件，按组件分块（`.exp-panel`、`.participant-card`、`.agent-panel` 等）。

**代理后端到非本机**：在启动前 `export PREFQUEST_BACKEND_URL=http://remote:8000`。

**排查"创建中…"卡住**：通常是后端挂了或端口不对。打开浏览器 DevTools Network，看 `/sessions` POST 是否 200。也可以直接访问 `http://localhost:8001/docs` 验证后端。
