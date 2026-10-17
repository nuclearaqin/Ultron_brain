"""神经符号智能体（目标 5/6）：规划-执行-观察-反思回路。

把因果图(符号)、嵌入(神经)、记忆、工具、推理器、世界模型、链接预测器
整合成一个真正的智能体：接收任务 → 检索上下文(RAG) → 规划工具调用 →
执行并观测 → 反思并(可能)修正 → 输出带推理痕迹的最终答复。
"""

from typing import Any, Dict, List, Optional

from .graph import CausalGraph
from .memory import MemoryStore
from .reasoner import Reasoner
from .tools import Toolbox


class NeuroSymbolicAgent:
    def __init__(self, graph: CausalGraph, embeddings, memory: MemoryStore,
                 reasoner: Reasoner, link_predictor=None, world_model=None):
        self.graph = graph
        self.embeddings = embeddings
        self.memory = memory
        self.reasoner = reasoner
        self.link_predictor = link_predictor
        self.world_model = world_model
        self.toolbox = Toolbox()
        self._register_tools()

    # ==================== 工具注册 ====================
    def _register_tools(self) -> None:
        tb = self.toolbox
        tb.register("query_effects", "查询某个原因会导致哪些后果", self._t_query_effects,
                    {"cause": "原因"})
        tb.register("query_causes", "查询某个结果由哪些原因引起", self._t_query_causes,
                    {"effect": "结果"})
        tb.register("find_path", "查找两个节点之间的因果路径", self._t_find_path,
                    {"cause": "原因", "effect": "结果"})
        tb.register("plan", "规划达成某目标所需的因果行动链", self._t_plan,
                    {"goal": "目标"})
        tb.register("monte_carlo", "用 test-time compute 估计因果把握", self._t_mc,
                    {"cause": "原因", "effect": "结果"})
        tb.register("intervene", "干预某原因观察下游影响", self._t_intervene,
                    {"cause": "原因"})
        tb.register("retrieve_memory", "检索历史记忆", self._t_memory,
                    {"query": "查询"})
        tb.register("propose_links", "链接预测：提议可能缺失的因果边", self._t_propose,
                    {"top_k": "数量"})
        tb.register("reflect", "反思并评估当前结论的可靠性", self._t_reflect,
                    {"question": "问题", "answer": "答案"})

    # ---- 工具实现 ----
    def _t_query_effects(self, cause: str) -> str:
        effs = self.graph.query_effects(cause)
        if not effs:
            return f"「{cause}」目前没有记录的后果。"
        return "；".join(f"{e}（{w:.2f}）" for e, w, _ in effs)

    def _t_query_causes(self, effect: str) -> str:
        causes = self.graph.query_causes(effect)
        if not causes:
            return f"「{effect}」目前没有记录的成因。"
        return "；".join(f"{c}（{w:.2f}）" for c, w, _ in causes)

    def _t_find_path(self, cause: str, effect: str) -> str:
        p = self.graph.find_path(cause, effect)
        if not p:
            return f"「{cause}」与「{effect}」之间没有因果路径。"
        return " → ".join(p) + f"（把握 {self.graph.path_strength(p):.2f}）"

    def _t_plan(self, goal: str) -> str:
        return self.reasoner.format_plan(goal)

    def _t_mc(self, cause: str, effect: str) -> str:
        return self.reasoner.self_consistent(cause, effect)

    def _t_intervene(self, cause: str) -> str:
        down = self.graph.intervene(cause)
        if not down:
            return f"干预「{cause}」后没有下游被牵动。"
        return "；".join(f"{k}→{v:.2f}" for k, v in sorted(down.items(), key=lambda x: -x[1]))

    def _t_memory(self, query: str) -> str:
        hits = self.memory.retrieve(query, k=3)
        if not hits:
            return "没有相关记忆。"
        return "\n".join(f"· {e['role']}：{e['text'][:80]}" for _, e in hits)

    def _t_propose(self, top_k: int = 5) -> str:
        if self.link_predictor is None or not self.link_predictor.trained:
            return "链接预测器尚未训练（先执行 /learn）。"
        nodes = self.graph.nodes()
        indexer = getattr(self.memory, "indexer", None)
        if indexer is not None and hasattr(indexer, "encode_many"):
            emb = indexer.encode_many(nodes)
        else:
            emb = self.embeddings.index_nodes(nodes)
        edges = {tuple(e) for e in self._edge_pairs()}
        props = self.link_predictor.propose(nodes, emb, edges, top_k=top_k)
        if not props:
            return "没有高于阈值的候选新边。"
        return "\n".join(f"· {a} → {b}（{p:.2f}）" for a, b, p in props)

    def _t_reflect(self, question: str, answer: str) -> str:
        score, note = self.reflect(question, answer)
        return f"自评可靠性 {score:.2f}：{note}"

    def _edge_pairs(self) -> List:
        return [(u, v) for u, v in self.graph.graph.edges()]

    # ==================== 反思 ====================
    def reflect(self, question: str, answer: str) -> tuple:
        """自评答案可靠性：检查答案引用的节点是否都真实存在于图中。"""
        score = 0.5
        note = ""
        nodes = self.graph.nodes()
        cited = [n for n in nodes if n and n in answer]
        if "无法" in answer or "没有记录" in answer or "不在" in answer:
            note = "答案暴露了知识缺口，依据不足，可靠性打折。"
            score = 0.3
        elif cited:
            # 引用越多的真实节点，越可信
            score = min(0.95, 0.5 + 0.15 * len(cited))
            note = f"答案引用了 {len(cited)} 个真实因果节点，证据较充分。"
        else:
            note = "答案未引用任何已知节点，可能为泛泛而谈。"
            score = 0.35
        self.memory.add_reflection(question, answer, score, note)
        return score, note

    # ==================== 意图解析 ====================
    def _intent(self, task: str) -> Dict[str, Any]:
        t = task
        # 匹配因果节点
        nodes = self.graph.nodes()
        matched = [n for n in nodes if n and n in t]
        return {"task": task, "matched": matched}

    # ==================== 主回路 ====================
    def run(self, task: str, max_steps: int = 6, verbose: bool = False) -> Dict[str, Any]:
        trace: List[str] = []
        # 1) RAG：检索上下文
        ctx = self.memory.build_context(task, self.graph)
        if verbose:
            trace.append("[检索] " + self.memory.format_context(ctx).replace("\n", " / "))

        # 2) 规划工具调用（意图路由）
        matched = self._intent(task)["matched"]
        steps: List[tuple] = []
        if len(matched) >= 2:
            steps.append(("find_path", {"cause": matched[0], "effect": matched[1]}))
        if any(k in task for k in ("为什么", "为何", "原因", "怎么来")):
            steps.append(("query_causes", {"effect": matched[0] if matched else task}))
        if any(k in task for k in ("会怎样", "导致", "后果", "影响", "会")):
            steps.append(("query_effects", {"cause": matched[0] if matched else task}))
        if any(k in task for k in ("计划", "规划", "达成", "怎么办", "如何达到")):
            steps.append(("plan", {"goal": matched[0] if matched else task}))
        if any(k in task for k in ("把握", "概率", "多可能", "test-time", "预测")):
            if len(matched) >= 2:
                steps.append(("monte_carlo", {"cause": matched[0], "effect": matched[1]}))
        if not steps:
            steps.append(("query_effects", {"cause": matched[0] if matched else task}))

        # 3) 执行 + 观察
        observations: List[str] = []
        for name, kwargs in steps[:max_steps]:
            tool = self.toolbox.get(name)
            if tool is None:
                continue
            try:
                obs = tool(**kwargs)
            except Exception as e:
                obs = f"工具 {name} 出错：{e}"
            observations.append(obs)
            trace.append(f"[执行] {name}({kwargs}) → {str(obs)[:120]}")

        # 4) 组装草稿答案
        answer = "\n".join(observations) if observations else "没有可执行的推演。"
        self.memory.add_episode("你", task)
        self.memory.add_episode("我", answer)

        # 5) 反思（可触发修正）
        score, note = self.reflect(task, answer)
        trace.append(f"[反思] 可靠性 {score:.2f}：{note}")
        if score < 0.4:
            answer = "（反思后提示：以下结论依据较弱，仅供参考）\n" + answer

        return {"task": task, "answer": answer, "trace": trace,
                "score": score, "context": ctx}
