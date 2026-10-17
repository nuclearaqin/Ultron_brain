"""推理层（目标 2）：思维链(CoT) + 目标驱动规划器 + test-time compute。

* chain_of_thought：把"因→果"的隐性图推理显式化为逐步推演。
* plan：给定目标(果)，反向/前向搜索出达成它的因果行动计划。
* test-time compute：把边权重当转移概率做蒙特卡洛采样——采样越多、估计越准，
  这就是"测试时计算"：推理阶段投入越多算力，结论越可靠。
"""

import random
from typing import Dict, List, Optional, Tuple

from .graph import CausalGraph


class Reasoner:
    def __init__(self, graph: CausalGraph, seed: int = 0):
        self.graph = graph
        self.rng = random.Random(seed)

    # ==================== 思维链 ====================
    def chain_of_thought(self, cause: str, effect: Optional[str] = None,
                         max_depth: int = 4) -> List[Tuple[str, str, float]]:
        """返回逐步推理链 [(node, 结论, 步强度), ...]。"""
        steps: List[Tuple[str, str, float]] = []
        if not self.graph.nodes():
            return steps
        if effect:
            path = self.graph.find_path(cause, effect)
            if path and len(path) >= 2:
                for a, b in zip(path[:-1], path[1:]):
                    w = self.graph.path_strength([a, b])
                    steps.append((a, b, w))
                return steps
        # 无目标：从 cause 出发逐步前向
        node = cause
        for _ in range(max_depth):
            effs = self.graph.query_effects(node)
            if not effs:
                break
            nxt, w, _ = effs[0]
            steps.append((node, nxt, w))
            node = nxt
        return steps

    def explain(self, cause: str, effect: Optional[str] = None,
                max_depth: int = 4) -> str:
        steps = self.chain_of_thought(cause, effect, max_depth)
        if not steps:
            return "无法从当前因果图推出推理链。"
        lines = []
        for i, (a, b, w) in enumerate(steps, 1):
            lines.append(f"  步骤{i}：「{a}」 → 「{b}」（把握 {w:.2f}）")
        return "\n".join(lines)

    # ==================== 规划器 ====================
    def plan(self, goal: str, max_steps: int = 6, top_k: int = 3,
             branch_limit: int = 3, max_plans: int = 50) -> List[dict]:
        """反向规划：从目标(果)向上找达成它的因果行动计划。

        大图剪枝：每节点只展开权重最高的 branch_limit 个前因、防环、
        达到 max_plans 条计划即提前终止，避免超大图上的指数爆炸。
        """
        plans: List[dict] = []

        def dfs(node, path, prob, visited):
            if len(plans) >= max_plans:
                return
            if len(path) > max_steps:
                return
            causes = self.graph.query_causes(node)
            # 只保留权重最高的 branch_limit 个前因
            causes = sorted(causes, key=lambda x: -x[1])[:branch_limit]
            if not causes:
                if len(path) >= 1:
                    plans.append({"path": path, "strength": prob})
                return
            for c, w, cond in causes:
                if c in visited:       # 防环
                    continue
                dfs(c, path + [{"action": c, "target": node,
                                "weight": w, "conditions": cond}],
                    prob * w, visited | {c})

        dfs(goal, [], 1.0, {goal})
        plans.sort(key=lambda p: -p["strength"])
        return plans[:top_k]

    def format_plan(self, goal: str, max_steps: int = 6) -> str:
        plans = self.plan(goal, max_steps)
        if not plans:
            return f"以当前因果图，找不到达成「{goal}」的行动链。"
        lines = [f"达成「{goal}」的候选计划（按把握排序）："]
        for i, p in enumerate(plans, 1):
            seq = " → ".join(f"[{s['action']}]" for s in reversed(p["path"]))
            lines.append(f"  计划{i}（把握 {p['strength']:.2f}）：{seq} ⇒ {goal}")
        return "\n".join(lines)

    # ==================== test-time compute ====================
    def _sample_rollout(self, cause: str, effect: str, max_steps: int = 8) -> bool:
        """把边权重当转移概率，做一次随机前向游走，看是否命中 effect。

        无条件边按原权重纳入；条件边在"条件未知"时以 0.5 折扣纳入，
        建模条件是否成立的不确定性，从而与思维链/规划器保持一致的语义。
        """
        node = cause
        for _ in range(max_steps):
            if node == effect:
                return True
            matched, unmet = self.graph._iter_edges(node, "out", None)
            options = [(v, d.get("weight", 1.0)) for _, v, d in matched]
            options += [(v, d.get("weight", 1.0) * 0.5) for _, v, d in unmet]
            if not options:
                return False
            names = [o[0] for o in options]
            weights = [o[1] for o in options]
            total = sum(weights)
            if total <= 0:
                return False
            r = self.rng.uniform(0, total)
            acc = 0.0
            node = names[-1]
            for name, w in zip(names, weights):
                acc += w
                if r <= acc:
                    node = name
                    break
        return node == effect

    def monte_carlo(self, cause: str, effect: str, samples: int = 200,
                    max_steps: int = 8) -> Dict:
        """test-time compute：蒙特卡洛估计 P(effect | cause)，样本越多越准。"""
        if not self.graph.find_path(cause, effect):
            return {"cause": cause, "effect": effect, "prob": 0.0,
                    "hits": 0, "samples": samples, "reachable": False}
        hits = sum(1 for _ in range(samples)
                   if self._sample_rollout(cause, effect, max_steps))
        return {"cause": cause, "effect": effect, "prob": hits / samples,
                "hits": hits, "samples": samples, "reachable": True}

    def self_consistent(self, cause: str, effect: str, samples: int = 200) -> str:
        """自洽性：用 test-time compute 给出"多大把握"。"""
        r = self.monte_carlo(cause, effect, samples)
        if not r["reachable"]:
            return f"「{cause}」推不到「{effect}」，无法给出把握估计。"
        return (f"test-time compute（{samples} 次采样）："
                f"「{cause}」导致「{effect}」的把握约 {r['prob']:.2f}"
                f"（命中 {r['hits']}/{r['samples']} 次）。")
