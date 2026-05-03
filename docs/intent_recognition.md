# 意图识别模块技术架构

## 一、定位与职责

将用户自然语言输入映射为结构化意图对象，供 LangGraph 编排层路由到对应处理节点。

意图识别包含两个子任务，需分开设计：

```
意图识别
├── 意图分类（Intent Classification）
│   └── 将用户表达归类到预定义类别：action / exhibit_qa / skill / ...
└── 实体识别（Slot Filling）
    └── 从用户表达中提取关键槽位：展品ID / 目的地 / 技能名称 / ...
```

**输入**：用户文本（经 ASR 或直接输入）
**输出**：`MultiIntent` 结构体，含意图类型、槽位、置信度、依赖关系

---

## 二、意图类型体系

| 类型 | 示例 | 处理路径 |
|------|------|---------|
| `action` | 点头、转身、过来 | 关键词拦截 → 机器人 SDK（< 50ms） |
| `exhibit_qa` | 这辆车是什么年代的？ | LLM → TTS |
| `chitchat` | 你叫什么名字？ | LLM → TTS |
| `skill` | 播放这辆车的视频 | LLM Function Call → Skill |
| `image_identify` | 这是什么车？（附图） | VLM → 展品识别 → LLM |
| `video_understand` | 看看你前面的环境 | 触发相机采集 → VLM → LLM |
| `qr_scan` | 扫码 | QR 解析 → Skill |
| `unknown` | 无法识别 | 提示用户重新表述 |

当前意图类型为 8 类，使用扁平分类体系（无需层级化）。如后续意图超过 20 类，需引入"一级大类 + 二级细类"的层级体系。

---

## 三、两级级联架构

```
用户输入
   │
   ▼
┌──────────────────────────────────┐
│  第一级：关键词快速拦截（< 1ms）   │
│  仅用于 action 类意图              │
│  词表有限、边界清晰、不需要 LLM    │
└──────────────────────────────────┘
   │ 命中 → 直接返回 action 意图
   │ 未命中 ↓
┌──────────────────────────────────┐
│  第二级：LLM 结构化分类（< 500ms）│
│  模型：qwen3-vl-flash-2026-01-22  │
│  输出：Pydantic structured output │
│  支持多意图（MultiIntent）         │
└──────────────────────────────────┘
   │ confidence ≥ 0.5 → 返回意图
   │ confidence < 0.5 → unknown
   ▼
返回 MultiIntent 对象
```

**为什么不用三级（去掉向量检索层）**

部分方案在关键词和 LLM 之间加入向量检索（Semantic Router）作为中间层。最终决定去掉：
- 当前意图类型只有 8 类，LLM 直接分类准确率已足够
- 需要持续维护 utterance 样本库，随展品更新持续扩张
- `exhibit_qa` 和 `chitchat` 语义边界模糊，向量匹配容易混淆
- 减少一个依赖，降低延迟链路复杂度

**置信度阈值说明**

阈值设为 0.5（而非更高的 0.75）：低于 0.5 说明模型自身判断模糊，强行路由反而误判；高于 0.5 的低置信度结果（0.5~0.75）仍可路由，response 节点可视情况追问。

---

## 四、数据结构

```python
class SingleIntent(BaseModel):
    type: Literal[
        "action", "exhibit_qa", "chitchat", "skill",
        "image_identify", "video_understand", "qr_scan", "unknown"
    ]
    confidence: float           # 0.0 ~ 1.0
    slots: dict[str, str]       # 提取的槽位
    action_name: str | None     # action 类专用，如 "nod", "wave"
    depends_on: int | None      # 多意图时，依赖第几个意图完成后执行

class MultiIntent(BaseModel):
    intents: list[SingleIntent] # 有序列表，支持一句话多意图
```

**槽位类型分类**

| 类型 | 示例 | 提取方式 |
|------|------|---------|
| 枚举型 | 展品ID、技能名称 | 关键词匹配 + LLM |
| 文本型 | 查询内容、问题描述 | LLM |
| 布尔型 | 确认/拒绝 | 关键词精确匹配 |
| 数值型 | 数量、编号 | 正则 + LLM |

**多意图示例**

输入："你过来，然后介绍一下这辆车"

```python
MultiIntent(intents=[
    SingleIntent(type="action", action_name="come_here", depends_on=None),
    SingleIntent(type="exhibit_qa", depends_on=0),  # 等机器人到位后执行
])
```

---

## 五、第一级：关键词拦截

来源：`config/intents.yaml` 的 `action_patterns` 字段，配置驱动。

```yaml
action_patterns:
  nod:       ["点头", "点点头"]
  shake:     ["摇头", "摇摇头"]
  come_here: ["过来", "来这里", "走过来"]
  stop:      ["停", "停下", "别动"]
  wave:      ["挥手", "招手"]
```

**实战注意点**

- 优先使用精确匹配，避免包含匹配误判（如"停一下看这个"被误拦截为 stop）
- 定期从对话日志中挖掘 ASR 误识别变体（如"转伸"→"转身"），补充到词表
- 关键词有歧义时不强行拦截，下沉到 LLM 判断

---

## 六、第二级：LLM 分类器

### Prompt 设计

Prompt 包含四个要素，零样本可达 90%+ 准确率，补充 3~5 条少样本示例后可达 95%+：

```
1. 角色定义：明确场景边界
   "你是汽车博物馆讲解机器人的意图识别专家"

2. 意图说明：清晰界定每类意图的范围和边界
   "exhibit_qa = 访客询问展品相关信息，包括历史、技术、参数等"
   "chitchat  = 与展品无关的闲聊，包括问候、问机器人自身等"

3. 输出约束：避免格式混乱
   "严格按 JSON 输出 MultiIntent 结构，不输出额外文字"

4. 少样本示例：补充典型及易混淆案例
   "这辆车什么年代的 → exhibit_qa"
   "你叫什么名字 → chitchat"
```

### LLM 调用约束

- **temperature = 0.1**：减少随机性，降低幻觉风险
- **锁定模型版本**：调用时指定 `qwen3-vl-flash-2026-01-22`，API 模型升级不影响效果
- **置信度 < 0.5**：归为 unknown，触发追问，不强行路由

### 上下文携带策略（CR 公式）

多轮对话时，不是携带历史越多越好。参考上下文相关性分数（Context Relevance）：

- **CR ≈ 1**（本轮与历史相关性适中）：携带前 2~3 轮
- **CR > 1**（历史是干扰）：仅用当前轮，不带历史
- **CR ≪ 1**（信息冗余）：减少携带轮数

博物馆讲解场景多为连续追问同一展品（CR 接近 1），携带 2~3 轮即可。动作指令场景（CR 通常 > 1）不带历史。

### 失败兜底

LLM 调用失败（网络/超时）时：
1. 降级到第一级关键词匹配
2. 关键词也未命中 → 返回 unknown
3. response 节点提示用户重新表述

---

## 七、与 LangGraph 的集成点

```
intent_router_node（LangGraph 节点）
    │
    ├── keyword_matcher.fast_match(text)    → 命中返回 action
    │
    └── llm_classifier.classify(text, ctx) → 返回 MultiIntent
```

`intent_router_node` 只负责调用，不含业务逻辑。
`keyword_matcher` 和 `llm_classifier` 独立于 LangGraph，可单独测试。

---

## 八、评测方案

测试集路径：`tests/eval/intent_cases.yaml`

```yaml
cases:
  - input: "点头"
    expected_type: action
    expected_action: nod
    level: keyword

  - input: "这辆车发动机多少排量？"
    expected_type: exhibit_qa
    level: llm

  - input: "你过来，然后介绍一下发动机"
    expected_types: [action, exhibit_qa]
    level: llm
    multi_intent: true
    expected_depends_on: {1: 0}
```

**意图分类验收标准**

| 指标 | 目标 |
|------|------|
| 关键词层准确率 | 100%（词表可穷举验证） |
| LLM 层准确率 | ≥ 90% |
| 多意图 depends_on 正确率 | ≥ 95% |
| P95 延迟（LLM 路径） | ≤ 500ms |
| P99 延迟（关键词路径） | ≤ 5ms |

**槽位提取验收标准**

使用严格 F1-score，以下三项同时满足才算正确：
1. 槽位类型正确（不混淆）
2. 槽位取值完整、格式归一化
3. 实体边界精准（无多取、漏取）

目标：必选槽位严格 F1 ≥ 0.90