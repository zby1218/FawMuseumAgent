# FawMuseumAgent

汽车博物馆讲解机器人 Agent 系统。

## 技术栈

| 模块 | 选型 |
|------|------|
| Agent 编排 | LangGraph 1.1.x |
| LLM / VLM | Qwen3-VL（DashScope API） |
| 包管理 | uv |

## 快速开始

**前置要求**：安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 克隆项目
git clone https://github.com/zby1218/FawMuseumAgent.git
cd FawMuseumAgent

# 安装依赖（含开发工具）
uv sync --extra dev

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

## 项目结构

```
FawMuseumAgent/
├── src/faw_museum/
│   ├── graph/          # LangGraph 状态机与节点
│   ├── intent/         # 意图识别
│   ├── skills/         # Skill 定义与执行
│   ├── memory/         # 记忆系统
│   └── config/         # 配置加载器
├── config/
│   └── intents.yaml    # 意图关键词配置
└── tests/
    ├── unit/           # 单元测试
    └── eval/           # 意图识别评测
```

## 开发

```bash
uv run pytest           # 运行测试
uv run ruff check src/  # 代码检查
```

## 设计文档

详见 [DESIGN.md](DESIGN.md)。