import re
from functools import lru_cache
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ..config.loader import load_llm_config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """去掉 Qwen3 的思考链标签及内容。"""
    return _THINK_RE.sub("", text).strip()


def _to_llama_messages(messages: list[BaseMessage]) -> list[dict]:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    result = []
    for m in messages:
        role = role_map.get(m.type, m.type)
        result.append({"role": role, "content": m.content})
    return result


class _LlamaCppQwen35(BaseChatModel):
    """llama-cpp-python (JamePeng fork) + Qwen35ChatHandler 的 LangChain 适配器。

    仅处理纯文本消息，image_url 内容不在 FawMuseumAgent 的使用范围内。
    """

    _llm: Any = None
    _temperature: float = 0.7
    _max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-qwen35"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        llama_msgs = _to_llama_messages(messages)
        resp = self._llm.create_chat_completion(
            messages=llama_msgs,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=False,
        )
        raw = resp["choices"][0]["message"]["content"] or ""
        text = _strip_think(raw)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        llama_msgs = _to_llama_messages(messages)
        raw_buf = ""
        in_think = False

        for chunk in self._llm.create_chat_completion(
            messages=llama_msgs,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        ):
            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content") or ""
            raw_buf += token

            # 过滤 <think>...</think> 块，只 yield 可见文本
            while True:
                if not in_think:
                    start = raw_buf.find("<think>")
                    if start == -1:
                        visible, raw_buf = raw_buf, ""
                        if visible:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(content=visible)
                            )
                        break
                    visible = raw_buf[:start]
                    if visible:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=visible)
                        )
                    raw_buf = raw_buf[start + 7:]
                    in_think = True
                else:
                    end = raw_buf.find("</think>")
                    if end == -1:
                        break
                    raw_buf = raw_buf[end + 8:]
                    in_think = False

        if raw_buf.strip():
            yield ChatGenerationChunk(message=AIMessageChunk(content=raw_buf.strip()))


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """返回配置好的 LLM 实例，供各节点直接调用。接口固定为 BaseChatModel。"""
    cfg = load_llm_config()
    provider = cfg.get("provider", "llama_cpp_qwen35")

    if provider == "llama_cpp_qwen35":
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen35ChatHandler

        handler = Qwen35ChatHandler(
            clip_model_path=cfg["mmproj_path"],
            enable_thinking=False,
            verbose=False,
        )
        llama = Llama(
            model_path=cfg["model_path"],
            chat_handler=handler,
            n_ctx=cfg.get("n_ctx", 2048),
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
            n_threads=cfg.get("n_threads", 16),
            verbose=False,
        )
        inst = _LlamaCppQwen35()
        inst._llm = llama
        inst._temperature = cfg.get("temperature", 0.7)
        inst._max_tokens = cfg.get("max_tokens", 512)
        return inst

    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg.get("api_key", "not-used"),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 512),
        )

    raise ValueError(f"不支持的 LLM provider: {provider!r}")