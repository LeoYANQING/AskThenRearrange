# Study 2 App 修改方案

**文档性质**：对 `study2_frontend_PRD.md`（v1.0）的增量修改说明  
**基准版本**：study2_frontend_PRD.md v1.0（2026-04-17）  
**修改版本**：v2.0（2026-04-20）  
**修改依据**：Study 2 设计讨论定稿（2026-04-20）

---

## 一、设计变更摘要

| # | 变更项 | 旧设计 | 新设计 |
|---|---|---|---|
| C1 | 对话预算 | 固定 B=6，自动结束 | 开放式，实验员手动结束 |
| C2 | 轮次计数器 | 参与者可见（"Question 3/6"） | 删除，不展示给任何人 |
| C3 | Phase 顺序 | …→ Prediction → Questionnaire | …→ Questionnaire → Prediction |
| C4 | Phase 2 参与者界面 | 参与者有独立屏幕显示 | Phase 2 无参与者屏幕；参与者只与机器人交互 |
| C5 | STT 触发方式 | 参与者按住麦克风按钮录音 | 实验员按住按钮录音，松开发送 |
| C6 | 每场景物品数 | 12 件（6 seen + 6 unseen） | 16 件（8 seen + 8 unseen） |
| C7 | 逐轮 PSR 记录 | 仅记录最终 PSR | 每轮结束后记录当前 PSR@k（用于学习曲线分析） |
| C8 | Phase 5 预测展示 | 参与者屏幕显示 | 实验员屏幕展示（转向参与者） |
| C9 | 配置字段 `budget` | 默认 6，可编辑 | 删除 |

---

## 二、逐模块修改说明

### 2.1 Experimenter Dashboard（实验员控制面板）

**新增控件**：

| 控件 | 位置 | 说明 |
|---|---|---|
| **End Dialogue** 按钮 | Phase 2 控制栏 | 实验员点击后结束对话，推进到 Phase 3（偏好表单）。参与者口头表示"可以结束了"后触发。 |
| **STT 录音按钮**（Press & Hold） | Phase 2 控制栏，每轮提交区 | 实验员按住录制参与者语音，松开自动发送至 Whisper STT；转录结果显示在实验员侧文本框，实验员可编辑后点击 Submit Answer。 |

**删除控件**：

- 删除 Budget remaining 显示
- 删除 Scene config 中的 `budget` 字段（`§3.2`）
- 删除 Session Control Panel 中的预算倒计时

**Phase 推进按钮顺序调整**（对应 C3）：

```
旧：Scene → Dialogue → Pref Form → Prediction → Questionnaire → Next Trial
新：Scene → Dialogue → Pref Form → Questionnaire → Prediction → Next Trial
```

**实时 PSR 监控**（新增，仅实验员可见）：

- 每次 Submit Answer 后，侧边栏实时显示当前 PSR@k（seen / unseen / total）
- 仅供实验员监控实验进展，参与者不可见

---

### 2.2 Participant UI（参与者界面）

**Phase 2（对话）**：参与者无屏幕交互。

- 删除原 Phase 2 的所有参与者界面元素（问题气泡、STT 录音按钮、文本输入框、预算计数器）
- Phase 2 期间参与者界面可显示空白等待画面，或关闭屏幕
- 机器人 TTS 为问题传递的唯一渠道

**Phase 1（场景展示）**：

- 保留参与者屏幕，显示 16 件物品 + 5 个容器的视觉网格
- 参与者按 "Ready" 后进入 Phase 2（屏幕可随即关闭或转为等待画面）

**Phase 3（偏好表单）** → **Phase 4（问卷）**（顺序不变，仅编号更新）：

- 通过平板/iPad 呈现，实验员在 Phase 2 结束后递给参与者
- 偏好表单：16 件物品全部拖放完成后方可提交（原 12 件改为 16 件）
- 问卷：NASA-TLX + PSC + Perceived Control，顺序不变，但此时参与者**尚未看到准确率**

**Phase 5（系统预测展示）**（原 Phase 4）：

- 改为在**实验员屏幕**上展示，实验员将屏幕转向参与者
- 显示内容不变：预测放置 vs 参与者偏好表单的对比，confirmed / inferred 标注，X/16 准确率摘要
- 参与者浏览后，进入退出访谈（最后一个 trial 结束后）或下一个 trial（前两个 trial 结束后）

---

### 2.3 Backend / API 修改

**删除**：

- `AgentState.budget_total` 字段（不再需要）
- `dialogue/next_question` 返回值中的 `budget_remaining` 字段
- 对话自动终止逻辑（原 B 轮后自动结束）

**新增**：

- `POST /dialogue/{session_id}/end` — 实验员手动结束对话，触发 Final Placement Planner，记录 `turn_count`
- `GET /dialogue/{session_id}/psr_curve` — 返回本 trial 截至当前的逐轮 PSR 序列（格式见下）

**修改**：

- `POST /dialogue/{session_id}/submit_answer` 返回值新增字段：
  ```json
  {
    "updated_state": {...},
    "psr_at_k": {
      "turn_index": 3,
      "seen_psr": 0.750,
      "unseen_psr": 0.625,
      "total_psr": 0.688
    }
  }
  ```

---

### 2.4 Data Model 修改

**`trial_start` 事件**：删除 `budget` 字段

```jsonc
// 旧
{ "type": "trial_start", ..., "budget": 6, ... }

// 新
{ "type": "trial_start", ..., "items_seen": [...8 items...], "items_unseen": [...8 items...], ... }
```

**`dialogue_turn` 事件**：新增 `psr_at_k` 字段

```jsonc
{
  "type": "dialogue_turn",
  "turn_index": 2,
  "pattern": "PE",
  "question": "...",
  "answer": "...",
  "stt_raw": "...",
  "answer_edited": false,
  "response_time_s": 9.4,
  "psr_at_k": { "seen": 0.750, "unseen": 0.625, "total": 0.688 },
  "timestamp": "..."
}
```

**`trial_end` 事件**（新增，替代原隐式结束）：

```jsonc
{
  "type": "trial_end",
  "trial_index": 0,
  "turn_count": 7,
  "termination_reason": "participant_ready",
  "timestamp": "..."
}
```

**事件顺序调整**（对应 C3）：

```
旧：dialogue_turn×N → preference_form → evaluation → questionnaire
新：dialogue_turn×N → trial_end → preference_form → questionnaire → evaluation
```

---

### 2.5 Scene Configuration 修改

**物品数量**：12 → 16（8 seen + 8 unseen）

```yaml
# config.yaml 示例（bedroom 场景）
scenes:
  - label: bedroom
    seen_items:
      [book, glasses, phone_charger, notebook, pen, earplugs, watch, journal]
    unseen_items:
      [lamp, headphones, charger_cable, hand_cream, eye_mask, alarm_clock, tissue_box, water_bottle]
    receptacles: [dresser, nightstand, closet, shelf, under_bed]
```

**Latin Square 更新**：物品列表调整后需重新验证场景配置的可行性（seen/unseen 分组是否合理）。

---

### 2.6 PrefQuest Core 集成修改

**state_initialization**（`§7.1`）：

```python
state: AgentState = {
    # 删除 "budget_total": 6
    "room": scene_label,
    "receptacles": receptacle_list,
    "seen_objects": items[:8],     # 8 seen
    "unseen_objects": items[8:],   # 8 unseen
    "qa_history": [],
    "confirmed_actions": [],
    "negative_actions": [],
    "confirmed_preferences": [],
    "negative_preferences": [],
    "unresolved_objects": list(items),
}
```

**终止逻辑**：对话终止由 `POST /dialogue/{session_id}/end` 触发（实验员手动），不再由预算耗尽自动触发。

---

## 三、不需要修改的部分

以下模块无需变更：

- TTS 集成（`§7.5`）
- STT 集成（`§7.6`）—— 触发方式从参与者侧移至实验员侧，但 API 调用逻辑不变
- PSR 计算函数（`§7.4`）—— 仅物品数量变化（12→16），公式不变
- Questionnaire 量表内容（NASA-TLX / PSC / Perceived Control）
- Session log 导出格式（JSONL）
- 非功能性需求（延迟、浏览器支持等）
- Final Ranking + Exit Comment（Phase 6，三个 trial 后）

---

## 四、Acceptance Criteria 更新

替换原 `§12` 中受影响的条目：

| # | 原条目 | 更新后 |
|---|---|---|
| AC-02 | Agent 生成问题，budget_remaining 被记录 | Agent 生成问题，pattern label (AO/PE/PI) 和 psr_at_k 被记录于每轮 |
| AC-05 | 偏好表单需 12 件全部完成 | 偏好表单需 16 件全部完成 |
| AC-06 | 正确计算 seen/unseen/total PSR | 正确计算 PSR，且逐轮 psr_at_k 序列完整记录 |
| AC-07 | JSONL 包含 dialogue_turn × B 等事件 | JSONL 包含 dialogue_turn × N（N 为实际轮次）+ trial_end 事件 |

**新增**：

| # | 条目 |
|---|---|
| AC-11 | 实验员按住 STT 按钮录音，松开后 3 s 内返回转录；实验员可编辑后提交 |
| AC-12 | End Dialogue 按钮触发 Final Placement Planner，记录 turn_count 和 trial_end 事件 |
| AC-13 | Phase 顺序为 Pref Form → Questionnaire → Prediction，问卷提交后方可展示预测 |
| AC-14 | psr_at_k 在每轮 submit_answer 后即时计算并写入 dialogue_turn 日志 |

---

*文档版本：v2.0 | 2026-04-20*
