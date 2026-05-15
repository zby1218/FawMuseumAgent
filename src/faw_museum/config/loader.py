# config/loader.py
# 统一读取 config/ 目录下的 YAML 配置文件，返回原始 dict。
# 调用方负责解释字段含义；本模块不做校验、不做缓存。

from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_intents() -> dict:
    # 意图定义与关键词表，供 router 和 content_filter 使用
    with open(_PROJECT_ROOT / "config" / "intents.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_llm_config() -> dict:
    # LLM 后端配置（provider / 模型路径 / 推理参数），供 llm/__init__.py 使用
    with open(_PROJECT_ROOT / "config" / "llm.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompts() -> dict:
    # 品牌身份与各节点业务约束，供节点构建 system message 使用
    with open(_PROJECT_ROOT / "config" / "system_prompts.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_blocked_keywords() -> dict:
    # 内容安全关键词表，供 content_filter 节点使用
    with open(_PROJECT_ROOT / "config" / "blocked_keywords.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)