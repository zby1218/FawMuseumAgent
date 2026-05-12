# Robot Voice Agent — LangGraph 架构文档

> 版本：v0.5  
> LangGraph：1.1.10  
> 状态：架构设计阶段，暂不接入 ASR/TTS，优先实现核心 graph 逻辑并支持流式输出测试

---

## 一、整体定位

本系统是 LiveKit Agents 框架中的 LLM 模块，接收 ASR 输出的文本，经过 LangGraph 状态图处理后，返回流式文本给 TTS。

测试阶段：直接输入文本字符串，观察流式 token 输出和 state 变化，不依赖 LiveKit。

```
[测试阶段]  文本输入 → LangGraph → 流式文本输出

[接入后]    ASR → LangGraph (LLM 插槽) → TTS → 语音
```

每台机器人对应一个独立的 `thread_id`（即 `robot_id`），通过 LangGraph checkpoint 隔离状态。每次用户说话触发一次 `graph.ainvoke()`，状态从上次 checkpoint 恢复，不重置。

### invoke 生命周期

系统中同一台机器人可能同时存在多个 invoke，它们相互独立、共享同一份 checkpoint：

```
invoke 1（动作任务）：dispatch → interrupt() → 挂起等待 ROS 回调
                                                    ↑
invoke 2（用户说话）：state_guard 读到 executing → discard → 结束（空输出）
invoke 3（用户说话）：state_guard 读到 executing → discard → 结束（空输出）
                                                    ↓
                              ROS完成 → Command(resume) → invoke 1 恢复 → TTS播放"我到了"
```

**关键设计决策**：执行期间（navigate / grasp / place）收到的所有 ASR 输入，在 State Guard 直接 discard，不走任何后续节点，不调 LLM，不产生输出。用户需等待当前动作完成后才能继续对话。

- invoke 1 挂起时不占用线程，仅为 checkpoint 中的一条记录
- invoke 2/3 走自己的流程，不影响 invoke 1 的挂起状态
- LangGraph checkpoint 有写锁，同一 thread_id 的并发写操作自动串行，不会数据错乱
- ROS 回调恢复 invoke 1 时，invoke 2/3 早已结束，TTS 输出无并发冲突

---

## 二、模块划分

```
robot_agent/
├── state.py          # State 定义（核心数据结构）
├── graph.py          # Graph 组装与编译入口
├── nodes/
│   ├── guard.py      # 模块1：状态守卫
│   ├── router.py     # 模块2：意图路由
│   ├── content_filter.py  # 模块3：内容安全过滤
│   ├── chitchat.py   # 模块4：闲聊
│   ├── knowledge.py  # 模块5：知识问答（RAG）
│   ├── intent.py     # 模块6：意图解析与槽位提取
│   ├── validate.py   # 模块7：槽位校验
│   ├── confirm.py    # 模块8：高风险确认（HITL）
│   ├── dispatch.py   # 模块9：ROS 指令分发
│   └── mute.py       # 模块10：静默控制
├── tools/
│   ├── ros_tools.py  # ROS service 封装为 LangChain tools
│   └── rag_tools.py  # RAG 检索工具
├── config/
│   ├── blocked_keywords.yaml   # 屏蔽关键词表（政治、黄赌毒、竞品品牌等）
│   └── system_prompts.yaml     # 各节点系统 prompt（含品牌身份与安全边界）
├── callback/
│   └── ros_callback.py  # ROS 回调处理（graph 外部，负责 resume invoke 1）
├── checkpointer.py   # Checkpoint 配置（测试用 Memory，生产用 SQLite）
├── streaming.py      # 流式输出适配（version="v2" 协议）
└── tts_queue.py      # TTS 串行队列（防止多 invoke 输出交叉）
```

---

## 三、State 设计

### 3.1 设计原则

- 字段只存跨轮次需要持久化的数据，不存可以实时查询的数据（如机器人位置、电量）
- 每个字段明确归属哪个节点写入、哪个节点读取
- 使用 `Annotated[list, add_messages]` 管理消息历史，替代手工维护的 history 列表
- 不存路由字段（如 `next_node`），路由逻辑全部在条件边函数里

### 3.2 State 定义

```python
from typing import Optional, Literal, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages


class SkillStep(TypedDict):
    skill: str        # 合法值由 dispatch 节点在 SUPPORTED_SKILLS 里校验
    params: dict      # {"destination": "厨房"} 等
    description: str  # 人类可读描述，用于日志和确认文本


class RobotState(TypedDict):

    # ── 会话标识 ──────────────────────────────────────────
    robot_id: str
    # 写入：初始化时
    # 读取：checkpointer 用作 thread_id，dispatch 用于区分机器人

    # ── 消息历史 ──────────────────────────────────────────
    messages: Annotated[list, add_messages]
    # 写入：每个节点均可追加，add_messages reducer 自动合并
    # 读取：router、chitchat、knowledge、intent_parse
    # 说明：包含 HumanMessage / AIMessage / ToolMessage

    # ── 系统状态（State Guard 读，各执行节点写）────────────
    silent_mode: bool
    # 写入：mute 节点（True）、wake_up 节点（False）
    # 读取：state_guard

    execution_status: Literal["idle", "executing", "waiting_input", "waiting_confirm"]
    # idle            — 空闲，正常接受输入
    # executing       — 正在执行 skill，新输入全部 discard
    # waiting_input   — 槽位不全，等待用户补充信息（validate 节点写入）
    # waiting_confirm — 高风险操作，等待用户二次确认（confirm 节点写入）
    # 写入：dispatch（executing）、validate（waiting_input）、
    #       confirm（waiting_confirm）、monitor/ros_callback（idle）
    # 读取：state_guard（单字段完成所有路由判断）

    current_task_type: Optional[str]
    # 写入：dispatch 节点（值与 SkillStep.skill 一致，合法性由 dispatch 保证）
    # 读取：state_guard（仅用于日志和 response_text 生成，不影响路由）
    # 重置：execution_status 回 idle 时清空

    # ── 意图分类（Router 写，Content Filter 条件边读）────
    intent_type: Optional[Literal["chitchat", "knowledge", "action", "mute"]]
    # 写入：router 节点
    # 读取：content_filter 条件边（放行后按此值路由到对应业务节点）
    # 重置：每次 invoke 开始时清空

    # ── 任务规划（intent_parse 写，dispatch/validate 读）───
    skill_sequence: list[SkillStep]
    # 写入：intent_parse 节点
    # 读取：validate、dispatch
    # 示例：[{skill: "navigate", params: {destination: "厨房"}},
    #        {skill: "dance",    params: {style: "wave"}},
    #        {skill: "action",   params: {name: "bow"}}]

    current_step: int
    # 写入：dispatch 节点（执行时递增）、intent_parse（重置为 0）
    # 读取：dispatch（取 skill_sequence[current_step]）

    # ── 执行结果（ROS 回调写，monitor 读）─────────────────
    last_ros_result: Optional[dict]
    # 写入：ros_callback（外部触发，resume graph 时注入）
    # 读取：monitor 节点（判断成功/失败/继续）
    # 示例：{"success": True, "message": "已到达目标位置"}

    # ── 输出 ──────────────────────────────────────────────
    response_text: str
    # 写入：所有有输出的节点
    # 读取：streaming 层，送给 LiveKit TTS（或测试时直接打印）
```

### 3.3 字段读写矩阵

| 字段 | 写入节点 | 读取节点 |
|------|---------|---------|
| `robot_id` | 初始化 | dispatch、checkpointer |
| `messages` | 所有节点 | router、chitchat、knowledge、intent_parse |
| `silent_mode` | mute、wake_up | state_guard |
| `execution_status` | dispatch（executing）、validate（waiting_input）、confirm（waiting_confirm）、ros_callback（idle） | state_guard |
| `current_task_type` | dispatch（值由 SUPPORTED_SKILLS 约束） | state_guard |
| `intent_type` | router | content_filter 条件边 |
| `skill_sequence` | intent_parse | validate、dispatch |
| `current_step` | dispatch、intent_parse | dispatch |
| `last_ros_result` | ros_callback | monitor |
| `response_text` | 所有有输出节点 | streaming 层 |

### 3.4 初始 State

```python
def create_initial_state(robot_id: str, user_input: str) -> RobotState:
    """仅在新会话第一次 invoke 时调用。
    后续多轮对话只传 {"messages": [HumanMessage(content=user_input)]}，
    state 从 checkpoint 恢复，不重新初始化。
    """
    return RobotState(
        robot_id=robot_id,
        messages=[HumanMessage(content=user_input)],
        silent_mode=False,
        execution_status="idle",
        current_task_type=None,
        intent_type=None,
        skill_sequence=[],
        current_step=0,
        last_ros_result=None,
        response_text="",
    )
```

---

## 四、节点职责说明

### 模块 1：状态守卫（guard.py）

**职责**：第一个执行的节点，纯逻辑判断，不调 LLM。读取系统状态字段，决定本次 invoke 走哪条路径。

**条件边输出**：

| 当前状态 | 路由目标 | 说明 |
|---------|---------|------|
| `silent_mode=True` + 检测到唤醒词 | `wake_up` | 关键词匹配，不调 LLM |
| `silent_mode=True` + 无唤醒词 | `discard` | 静默期间丢弃所有输入 |
| `execution_status=executing` | `discard` | 执行动作期间丢弃所有输入，等待 ROS 完成 |
| `execution_status=waiting_input` | `intent_parse` | 用户补充了槽位信息，重新解析 |
| `execution_status=waiting_confirm` | `handle_confirmation` | 用户确认/拒绝高风险操作 |
| `execution_status=idle` | `router` | 正常意图识别流程 |

**设计说明**：执行期间（navigate / grasp / place）统一 discard，不区分任务类型。用户需等动作完成后才能继续对话，简化了并发处理逻辑，避免 TTS 输出交叉。

**注意**：唤醒词检测用关键词匹配，不调 LLM，延迟接近零。

---

### 模块 2：意图路由（router.py）

**职责**：调 LLM 做意图分类，输出意图类型。这是系统第一次调云端 LLM。

**输出意图类型**：`chitchat` / `knowledge` / `action` / `mute`

**条件边输出**：所有意图统一先进 `content_filter`，由 content_filter 决定放行或拦截。

| 意图 | 路由目标 |
|-----|---------|
| `chitchat` / `knowledge` / `action` / `mute` | `content_filter` |

---

### 模块 3：内容安全过滤（content_filter.py）

**职责**：两层安全审核，拦截不允许回答的内容。位于 Router 之后、所有业务节点之前，是 graph 的安全闸门。

**Graph 位置**：

```
State Guard → Router → Content Filter → chitchat / knowledge / intent_parse / mute
                              ↓
                         拦截 → 固定拒绝回复 → END
```

**第一层：关键词预筛（不调 LLM，零延迟）**

维护一份可热更新的关键词表（`config/blocked_keywords.yaml`），按类别分组：

```yaml
# config/blocked_keywords.yaml
political:
  - 关键词列表...
gambling_drug:
  - 关键词列表...
competitor_brands:
  - 比亚迪
  - 蔚来
  - 小鹏
  - 理想
  - 特斯拉
  - 宝马
  - 奔驰
  - 奥迪
  # ... 持续维护
competitor_comparison:
  - "哪个好"
  - "对比"
  - "比较"
  - "怎么样"     # 需配合竞品品牌名同时出现才触发
```

**匹配规则**：

- 政治 / 黄赌毒类：单一关键词命中即拦截
- 竞品类：需要竞品品牌名 + 比较类动词同时出现才拦截（避免"我之前开特斯拉，现在想了解一汽"被误杀）

**第二层：系统 prompt 约束（LLM 调用时生效）**

所有调 LLM 的节点（router、chitchat、knowledge、intent_parse）共享统一的系统 prompt，定义品牌身份和安全边界：

```yaml
# config/system_prompts.yaml
brand_identity: |
  你是一汽集团的智能机器人助手。
  你必须遵守以下规则：
  1. 不讨论政治、宗教、赌博、毒品相关话题
  2. 不评价、比较任何其他汽车品牌或竞品
  3. 被问到竞品时，礼貌引导话题到一汽自身产品
  4. 不发表任何可能引起争议的政治观点
  5. 涉及国家政策时，只陈述官方公开信息，不做评价和解读
  6. 遇到无法回答的问题，回复"这个问题我不太方便回答，有什么其他可以帮您的吗？"
```

**第一层和第二层的分工**：

| 场景 | 第一层（关键词） | 第二层（系统 prompt） |
|------|:---:|:---:|
| "讲个黄色笑话" | 拦截 | — |
| "比亚迪和一汽哪个好" | 拦截（竞品+比较同时命中） | — |
| "特斯拉最近怎么样" | 拦截（竞品+评价） | — |
| "我之前开的特斯拉，一汽有什么推荐" | 放行（竞品出现但无比较意图） | LLM 自动聚焦一汽产品 |
| "你觉得中美关系会怎么发展" | 可能漏过 | LLM 按系统 prompt 拒答 |
| "你觉得某某领导人怎么样" | 可能漏过 | LLM 按系统 prompt 拒答 |

**拦截回复**：

```python
BLOCK_RESPONSES = {
    "political": "这个话题我不太方便讨论，有什么其他可以帮您的吗？",
    "gambling_drug": "抱歉，我无法回答这类问题。有什么其他可以帮您的吗？",
    "competitor": "我对其他品牌不太了解，不过我可以介绍一下我们一汽的产品，您感兴趣吗？",
}
```

**条件边输出**：

| 结果 | 路由目标 |
|------|---------|
| 拦截（关键词命中） | 直接写 `response_text` → `END` |
| 放行 | 按 Router 识别的意图类型路由到对应节点 |

**State 影响**：content_filter 不新增 state 字段。拦截时直接写 `response_text` 并结束。放行时透传 Router 的意图分类结果。意图类型暂存在 `messages` 中作为 metadata，或通过条件边函数闭包传递。

---

### 模块 4：闲聊（chitchat.py）

**职责**：LLM 直接生成回复，流式输出，graph 结束。不涉及任何工具调用。系统 prompt 中包含品牌身份约束（第二层安全防线）。

---

### 模块 5：知识问答（knowledge.py）

**职责**：RAG 检索 + LLM 生成回复，流式输出，graph 结束。

**子模块**（`tools/rag_tools.py`）：
- 向量检索（当前 mock，后续接入实际知识库）
- 结果重排
- LLM 基于检索结果生成答案

---

### 模块 6：意图解析（intent.py）

**职责**：调 LLM 将用户指令拆解为 `skill_sequence`，提取每个 skill 的槽位参数。

**输入**：`messages` 中的最新用户指令  
**输出**：写入 `skill_sequence`，`current_step=0`

**示例**：
```
输入："去厨房拿杯子放到桌上"
输出：[
  {skill: navigate, params: {destination: 厨房}, description: 导航到厨房},
  {skill: grasp,    params: {object: 杯子},      description: 抓取杯子},
  {skill: place,    params: {location: 桌上},    description: 放置到桌上},
]
```

---

### 模块 7：槽位校验（validate.py）

**职责**：检查 `skill_sequence` 中每个 skill 的必要参数是否完整，不调 LLM，纯规则校验。

**条件边输出**：
- 缺少必要槽位 → 生成追问文本，写入 `response_text`，graph 结束（等下一轮补充）
- 校验通过 → `confirm`

---

### 模块 8：高风险确认（confirm.py）

**职责**：判断当前 `skill_sequence` 是否属于高风险操作，若是则暂停等待用户确认。

**HITL 实现**：使用 LangGraph 1.1 的 `interrupt()` 原语暂停 graph，保存 checkpoint，等用户下一轮输入后通过 `Command(resume=...)` 恢复。

**条件边输出**：
- 高风险 → 触发 `interrupt()`，返回确认问句
- 非高风险 → `dispatch`

---

### 模块 9：ROS 指令分发（dispatch.py）

**职责**：取 `skill_sequence[current_step]`，调对应 ROS tool，设置执行状态，返回即时回复，然后用 `interrupt()` 挂起 graph 等待 ROS 回调。

**执行流**：

```
dispatch 节点
  1. 读取 skill_sequence[current_step]
  2. 调用 ros_tools.py 中对应的 tool
  3. 写入 execution_status=executing，current_task_type
  4. 写入 response_text（如"好的，正在去厨房"）→ TTS 立即播放
  5. interrupt() → graph 挂起，保存 checkpoint
                        ↓
              （invoke 1 挂起，进程继续处理其他事情）
                        ↓
              ROS 任务完成 → ros_callback.py 收到通知
                        ↓
              graph.ainvoke(Command(resume=result), config)
                        ↓
monitor 节点（dispatch 之后）
  - 读取 last_ros_result
  - 成功 + 还有下一步 → current_step += 1 → 回到 dispatch
  - 成功 + 全部完成  → execution_status=idle → response_text="任务完成"
  - 失败             → execution_status=idle → response_text="执行失败：原因"
```

**注意**：`interrupt()` 挂起的是 invoke 1，不阻塞进程。挂起期间收到的新 ASR 输入作为独立的 invoke 2 进来，在 State Guard 被 discard，不影响 invoke 1 的恢复。

---

### 模块 10：静默控制（mute.py）

**职责**：处理进入/退出静默模式。

- `mute` 节点：设置 `silent_mode=True`，调音量接口设为 0，返回"好的，需要时叫我"
- `wake_up` 节点：设置 `silent_mode=False`，恢复音量，返回"我在呢"
- `discard` 节点：返回空字符串，不触发 TTS

---

## 五、流式输出与 TTS 队列

### 5.1 流式输出

使用 LangGraph 1.1 的 `stream_mode="messages"` 协议，token 级流式仅在 `chitchat` 和 `knowledge` 节点启用。其他节点输出固定文本，直接写 `response_text`，不需要流式。

```python
async def run_streaming(graph, input_state, config):
    async for chunk in graph.astream(input_state, config, stream_mode="messages"):
        message_chunk, metadata = chunk
        if hasattr(message_chunk, "content") and message_chunk.content:
            await tts_queue.put(message_chunk.content)
            # 测试阶段：print(message_chunk.content, end="", flush=True)
```

### 5.2 TTS 串行队列（tts_queue.py）

**问题**：invoke 1（ROS 回调恢复后输出"我到了"）和 invoke 2（闲聊回复）理论上可能同时往 TTS 推内容，造成语音交叉。

**解法**：所有节点的输出统一 put 进一个 `asyncio.Queue`，单一 consumer 串行消费，保证 TTS 输出有序。

```python
# tts_queue.py
import asyncio

tts_queue: asyncio.Queue[str] = asyncio.Queue()

async def tts_consumer():
    """单一 consumer，串行播放，防止语音交叉"""
    while True:
        text = await tts_queue.get()
        if text:  # discard 节点输出空字符串，跳过
            await tts.speak(text)  # 接入 LiveKit 时替换
            # 测试阶段：print(f"[TTS] {text}")
```

**实际上**，由于执行期间所有 ASR 输入都被 discard，invoke 1 恢复时不会有其他 invoke 在产生输出，TTS 队列在当前架构下几乎不会遇到并发冲突，但作为防御性设计保留。

---

## 六、测试方案（不接 ASR/TTS）

### 6.1 单轮测试入口

```python
async def test_input(robot_id: str, user_text: str):
    config = {"configurable": {"thread_id": robot_id}}
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=user_text)]},
        config,
        stream_mode="messages"
    ):
        message_chunk, _ = chunk
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print()
```

### 6.2 模拟 ROS 回调

```python
async def simulate_ros_complete(robot_id: str, success: bool, message: str):
    """模拟 ROS 任务完成，触发 invoke 1 恢复"""
    config = {"configurable": {"thread_id": robot_id}}
    result = {"success": success, "message": message}
    async for chunk in graph.astream(
        Command(resume=result),
        config,
        stream_mode="messages"
    ):
        message_chunk, _ = chunk
        if message_chunk.content:
            print(f"[ROS恢复] {message_chunk.content}", end="", flush=True)
    print()
```

### 6.3 多轮测试场景

```python
# 场景1：正常动作任务，执行期间输入被丢弃
await test_input("robot_01", "去厨房拿杯子放到桌上")
# → 输出"好的，正在去厨房"，graph 挂起

await test_input("robot_01", "你好啊")
# → 执行中，discard，无输出

await simulate_ros_complete("robot_01", True, "已到达厨房")
# → invoke 1 恢复，自动 dispatch 下一步 grasp，输出"到厨房了，正在抓杯子"

await simulate_ros_complete("robot_01", True, "抓取成功")
# → dispatch place，输出"抓到了，正在放桌上"

await simulate_ros_complete("robot_01", True, "放置成功")
# → 全部完成，输出"好了，杯子放到桌上了"，execution_status=idle

# 场景2：静默模式
await test_input("robot_01", "你安静一会儿")   # → "好的，需要时叫我"
await test_input("robot_01", "帮我拿个东西")   # → discard，无输出
await test_input("robot_01", "咱们再聊聊天")   # → "我在呢，有什么事？"

# 场景3：内容安全过滤
await test_input("robot_01", "比亚迪和一汽哪个好")
# → content_filter 拦截，"我对其他品牌不太了解，不过我可以介绍一下我们一汽的产品"

await test_input("robot_01", "我之前开的特斯拉，一汽有什么推荐")
# → content_filter 放行（竞品出现但无比较意图），LLM 聚焦一汽产品回复

# 场景4：高风险确认
await test_input("robot_01", "删除所有导航点")  # → 触发 interrupt，"确定要删除吗？"
await test_input("robot_01", "确定")            # → Command(resume=True)，执行操作
```

---

## 七、待定事项（后续细化）

- [ ] RAG 知识库具体内容和检索方式（向量库选型）
- [ ] 高风险操作的判断规则（哪些 skill 触发 confirm 节点）
- [ ] ROS 回调的具体触发方式（HTTP callback / ROS topic / 消息队列）
- [ ] Checkpoint 持久化方案（测试用 InMemorySaver，生产用 AsyncSqliteSaver 或 Redis）
- [ ] 多机器人并发时 checkpointer 隔离验证
- [ ] 唤醒词列表维护方式
- [ ] monitor 节点失败重试策略（重试几次、重试间隔）
- [ ] dispatch 和 monitor 是否合并为同一节点
- [ ] `blocked_keywords.yaml` 完整词表整理（政治、黄赌毒、竞品品牌清单）
- [ ] 竞品关键词匹配规则细化（组合匹配逻辑、误杀率评估）
- [ ] `system_prompts.yaml` 品牌身份 prompt 调优（与一汽品牌方确认措辞）
