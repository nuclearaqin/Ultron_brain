"""工具注册表（目标 5 的工具使用部分）。

Toolbox 维护"名字 -> 工具"的注册表，提供 describe()（供路由/自省）与
route()（用嵌入相似度把自然语言任务路由到最合适的工具）。
"""

from typing import Any, Callable, Dict, List, Optional


class Tool:
    def __init__(self, name: str, description: str, fn: Callable,
                 params: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.fn = fn
        self.params = params or {}

    def __call__(self, **kwargs) -> Any:
        return self.fn(**kwargs)


class Toolbox:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, fn: Callable,
                 params: Optional[Dict[str, str]] = None) -> None:
        self.tools[name] = Tool(name, description, fn, params)

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def names(self) -> List[str]:
        return list(self.tools.keys())

    def describe(self) -> str:
        return "\n".join(f"  · {n}：{t.description}" for n, t in self.tools.items())

    def descriptions(self) -> Dict[str, str]:
        return {n: t.description for n, t in self.tools.items()}

    def route(self, query: str, embeddings) -> Optional[str]:
        """按语义相似度把 query 路由到最合适的工具名。"""
        names = list(self.tools.keys())
        if not names:
            return None
        descs = [self.tools[n].description for n in names]
        q = embeddings.encode([query])
        d = embeddings.encode(descs)
        sims = embeddings.similarity(q, d).flatten()
        return names[int(sims.argmax())]
