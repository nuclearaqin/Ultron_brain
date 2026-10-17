"""记忆 + 检索增强 + 反思回路（目标 5 的记忆/检索部分）。

三种记忆：
  * 情景记忆(episodic)：对话/事件记录，带嵌入，可语义检索；
  * 语义记忆(semantic)：因果图中的事实（引用外部 CausalGraph）；
  * 程序记忆(procedural)：工具清单（引用外部 Toolbox）。

RAG：retrieve(query) 召回相关情景 + 相关因果事实，build_context 拼成
可用于作答/反思的上下文。
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MemoryStore:
    def __init__(self, embeddings, max_episodes: int = 500, indexer=None):
        self.embeddings = embeddings          # EmbeddingModel
        self.indexer = indexer                # 可选：共享的持久化节点嵌入索引
        self.episodes: List[Dict[str, Any]] = []
        self.ep_emb: List[np.ndarray] = []
        self.max_episodes = max_episodes
        self.reflections: List[Dict[str, Any]] = []

    # ==================== 情景记忆 ====================
    def add_episode(self, role: str, text: str, tags: Optional[List[str]] = None) -> None:
        text = (text or "")[:1000]
        if not text.strip():
            return
        self.episodes.append({
            "role": role, "text": text, "tags": tags or [],
            "ts": time.time(),
        })
        self.ep_emb.append(self.embeddings.encode([text])[0])
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
            self.ep_emb.pop(0)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, Dict]]:
        """语义检索最相关的情景记忆。"""
        if not self.episodes:
            return []
        q = self.embeddings.encode([query])
        sims = self.embeddings.similarity(q, np.array(self.ep_emb)).flatten()
        ranked = sorted(zip(sims.tolist(), self.episodes), key=lambda x: -x[0])
        return [(s, e) for s, e in ranked[:k]]

    # ==================== 反思回路 ====================
    def add_reflection(self, question: str, answer: str, score: float,
                       note: str = "") -> None:
        self.reflections.append({
            "question": question, "answer": answer, "score": score,
            "note": note, "ts": time.time(),
        })
        if len(self.reflections) > 200:
            self.reflections.pop(0)

    def last_reflection(self) -> Optional[Dict]:
        return self.reflections[-1] if self.reflections else None

    # ==================== RAG 上下文组装 ====================
    def build_context(self, query: str, graph=None, k_ep: int = 3,
                      k_fact: int = 4) -> Dict[str, Any]:
        """组装 RAG 上下文：相关情景 + 相关因果事实。"""
        ctx: Dict[str, Any] = {"episodes": [], "facts": []}
        for _, e in self.retrieve(query, k_ep):
            ctx["episodes"].append(f"{e['role']}：{e['text'][:120]}")
        if graph is not None:
            nodes = graph.nodes()
            if nodes:
                if self.indexer is not None and hasattr(self.indexer, "encode_many"):
                    emb = self.indexer.encode_many(nodes)
                else:
                    emb = self.embeddings.index_nodes(nodes)
                matches = self.embeddings.best_matches(query, nodes, emb, k=k_fact)
                for node, sim in matches:
                    effs = graph.query_effects(node)[:2]
                    causes = graph.query_causes(node)[:2]
                    fact = {"node": node, "sim": round(sim, 3),
                            "effects": [e for e, _, _ in effs],
                            "causes": [c for c, _, _ in causes]}
                    ctx["facts"].append(fact)
        return ctx

    def format_context(self, ctx: Dict[str, Any]) -> str:
        lines = []
        if ctx.get("episodes"):
            lines.append("相关记忆：\n  " + "\n  ".join(ctx["episodes"]))
        if ctx.get("facts"):
            parts = []
            for f in ctx["facts"]:
                parts.append(f"「{f['node']}」→ 果:{f['effects']} / 因:{f['causes']}")
            lines.append("相关因果：\n  " + "\n  ".join(parts))
        return "\n".join(lines) if lines else "（无相关记忆）"
