"""因果图（符号层）：networkx 有向图 + 持久化 + 条件因果 + 干预/反事实。

这是"神经-符号"中的符号半边。承载多因多果、条件因果、权重、路径、
do-干预、反事实推理，并做加载期数据清洗（剔除空节点/命令/超长污染节点）。
"""

import json
import os
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .config import MAX_NODE_LEN, TIME_ALIAS


# 边级元数据字段白名单：参与 save/load 序列化（来源可追溯）
EDGE_META_KEYS = ("source", "relation", "raw_weight")


class CausalGraph:
    def __init__(self, storage_path: str = "causal_graph.json"):
        self.storage_path = storage_path
        self.graph = nx.DiGraph()
        self._lock = threading.RLock()
        self._condition_vocab: Dict[str, Tuple[str, str]] = {}   # 别名 -> (键, 规范值)

    # ==================== 加载 / 保存 ====================
    @staticmethod
    def _is_valid_node(n: str) -> bool:
        n = (n or "").strip()
        if not n:
            return False
        if len(n) > MAX_NODE_LEN:
            return False
        if n in ("EXIT", "/exit") or n.startswith("/"):
            return False
        return True

    def load(self) -> bool:
        """从磁盘加载；文件缺失返回 False。加载时剔除污染节点。"""
        if not os.path.exists(self.storage_path):
            return False
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return False
        with self._lock:
            kept = [n for n in data.get("nodes", []) if self._is_valid_node(n)]
            kept_set = set(kept)
            self.graph.clear()
            self.graph.add_nodes_from(kept)
            for edge in data.get("edges", []):
                u, v = edge.get("from"), edge.get("to")
                if not self._is_valid_node(u) or not self._is_valid_node(v):
                    continue
                if u not in kept_set or v not in kept_set:
                    continue
                meta = {k: edge[k] for k in EDGE_META_KEYS if k in edge}
                self.graph.add_edge(u, v,
                                    weight=edge.get("weight", 1.0),
                                    conditions=edge.get("conditions", {}),
                                    **meta)
            self._refresh_condition_vocab()
        return True

    def save(self) -> None:
        with self._lock:
            data = {
                "version": 2,
                "nodes": list(self.graph.nodes()),
                "edges": [
                    {
                        "from": u, "to": v,
                        "weight": d.get("weight", 1.0),
                        "conditions": d.get("conditions", {}),
                        **{k: d[k] for k in EDGE_META_KEYS if k in d},
                    }
                    for u, v, d in self.graph.edges(data=True)]}
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ==================== 结构操作 ====================
    def add_causal_link(self, cause: str, effect: str, weight: float = 1.0,
                        conditions: Optional[Dict] = None,
                        overwrite: bool = False,
                        meta: Optional[Dict] = None) -> bool:
        """添加/覆盖一条因果边。

        overwrite=False（默认）：若边已存在则**不改动**并返回 False（防止
        交互教学静默覆盖掉已有权重/条件）；overwrite=True 才允许重写。
        meta：边级元数据（source/relation/raw_weight 等，随 save/load 持久化）。
        返回是否实际写入。
        """
        cause, effect = cause.strip(), effect.strip()
        if not cause or not effect or cause == effect:
            return False
        with self._lock:
            if self.graph.has_edge(cause, effect) and not overwrite:
                return False
            self.graph.add_node(cause)
            self.graph.add_node(effect)
            attrs: Dict = {"weight": weight, "conditions": conditions or {}}
            if meta:
                attrs.update({k: v for k, v in meta.items() if k in EDGE_META_KEYS})
            self.graph.add_edge(cause, effect, **attrs)
            self._refresh_condition_vocab()
        return True

    def edge_info(self, cause: str, effect: str) -> Optional[Dict]:
        """返回已有边的 (weight, conditions)；不存在返回 None。"""
        with self._lock:
            if self.graph.has_edge(cause, effect):
                return dict(self.graph[cause][effect])
        return None

    def bulk_add_causal_links(self, edges) -> int:
        """批量导入边，只刷新一次条件词表（避免逐条 O(E) 重建）。

        edges: [(cause, effect, weight, conditions[, meta]), ...] 迭代器或列表，
        第 5 元素 meta 为可选的边级元数据 dict（source/relation/raw_weight）。
        返回实际新增的边数。
        """
        added = 0
        with self._lock:
            for row in edges:
                cause, effect, weight, conditions = row[:4]
                meta = row[4] if len(row) > 4 and row[4] else None
                cause, effect = cause.strip(), effect.strip()
                if not cause or not effect or cause == effect:
                    continue
                if self.graph.has_edge(cause, effect):
                    continue
                self.graph.add_node(cause)
                self.graph.add_node(effect)
                attrs: Dict = {"weight": weight, "conditions": conditions or {}}
                if meta:
                    attrs.update({k: v for k, v in meta.items() if k in EDGE_META_KEYS})
                self.graph.add_edge(cause, effect, **attrs)
                added += 1
            self._refresh_condition_vocab()
        return added

    def remove_causal_link(self, cause: str, effect: str) -> bool:
        with self._lock:
            if self.graph.has_edge(cause, effect):
                self.graph.remove_edge(cause, effect)
                self._refresh_condition_vocab()
                return True
        return False

    def set_edge_weight(self, cause: str, effect: str, weight: float) -> bool:
        with self._lock:
            if self.graph.has_edge(cause, effect):
                self.graph[cause][effect]["weight"] = weight
                return True
        return False

    def has_edge(self, u: str, v: str) -> bool:
        with self._lock:
            return self.graph.has_edge(u, v)

    # ==================== 条件因果 ====================
    def _refresh_condition_vocab(self) -> None:
        vocab: Dict[str, Tuple[str, str]] = {}
        for _, _, d in self.graph.edges(data=True):
            for key, val in (d.get("conditions") or {}).items():
                vocab[val] = (key, val)
                if key == "时间":
                    for alias, canon in TIME_ALIAS.items():
                        if canon == val:
                            vocab[alias] = (key, val)
        self._condition_vocab = vocab

    def extract_context(self, text: str) -> Dict[str, str]:
        ctx: Dict[str, str] = {}
        with self._lock:
            vocab = dict(self._condition_vocab)
        for token, (key, val) in vocab.items():
            if token and token in text:
                ctx[key] = val
        return ctx

    # ==================== 查询 ====================
    def nodes(self) -> List[str]:
        with self._lock:
            return list(self.graph.nodes())

    def edge_count(self) -> int:
        with self._lock:
            return self.graph.number_of_edges()

    def node_count(self) -> int:
        with self._lock:
            return self.graph.number_of_nodes()

    def _iter_edges(self, node: str, direction: str,
                    context: Optional[Dict]) -> Tuple[List, List]:
        """direction: 'out' 因->果 / 'in' 果<-因。返回 (匹配, 不匹配)。"""
        rows = []
        with self._lock:
            if not self.graph.has_node(node):
                return [], []
            if direction == "out":
                for succ in self.graph.successors(node):
                    d = dict(self.graph[node][succ])
                    rows.append((node, succ, d))
            else:
                for pred in self.graph.predecessors(node):
                    d = dict(self.graph[pred][node])
                    rows.append((pred, node, d))
        ctx = context or {}
        matched, unmet = [], []
        for u, v, d in rows:
            cond = d.get("conditions") or {}
            ok = all(ctx.get(k) == val for k, val in cond.items())
            (matched if ok else unmet).append((u, v, d))
        matched.sort(key=lambda x: -x[2].get("weight", 1.0))
        unmet.sort(key=lambda x: -x[2].get("weight", 1.0))
        return matched, unmet

    def query_effects(self, cause: str, context: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        matched, _ = self._iter_edges(cause, "out", context)
        return [(v, d.get("weight", 1.0), d.get("conditions", {})) for _, v, d in matched]

    def query_causes(self, effect: str, context: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        matched, _ = self._iter_edges(effect, "in", context)
        return [(u, d.get("weight", 1.0), d.get("conditions", {})) for u, _, d in matched]

    def find_path(self, cause: str, effect: str) -> Optional[List[str]]:
        with self._lock:
            if self.graph.has_node(cause) and self.graph.has_node(effect):
                try:
                    return nx.shortest_path(self.graph, cause, effect)
                except nx.NetworkXNoPath:
                    return None
        return None

    def path_strength(self, path: List[str]) -> float:
        p = 1.0
        with self._lock:
            for a, b in zip(path[:-1], path[1:]):
                if not self.graph.has_edge(a, b):
                    return 0.0
                p *= self.graph[a][b].get("weight", 1.0)
        return p

    def reachable_chains(self, node: str, max_len: int = 3,
                         limit: int = 3) -> List[Tuple[List[str], float]]:
        """从 node 出发的代表性因果链 [(路径, 乘积强度)]。"""
        with self._lock:
            if not self.graph.has_node(node):
                return []
            chains = []
            stack = [(node, [node], 1.0)]
            while stack:
                cur, path, prob = stack.pop()
                if len(path) >= 2:
                    chains.append((list(path), prob))
                if len(path) >= max_len:
                    continue
                succs = list(self.graph.successors(cur))
                succs.sort(key=lambda s: -self.graph[cur][s].get("weight", 1.0))
                for s in succs[:2]:
                    w = self.graph[cur][s].get("weight", 1.0)
                    stack.append((s, path + [s], prob * w))
        chains.sort(key=lambda x: -x[1])
        return chains[:limit]

    # ==================== 干预与反事实 ====================
    def intervene(self, cause: str, value: str = "发生") -> Dict[str, Any]:
        """do(cause)：切断其所有入边后传播，观察下游激活概率（近似）。"""
        with self._lock:
            if not self.graph.has_node(cause):
                return {"error": "节点不存在"}
            in_edges = list(self.graph.in_edges(cause, data=True))
            self.graph.remove_edges_from(in_edges)
            activation = {cause: 1.0}
            queue = deque([cause])
            while queue:
                current = queue.popleft()
                cur_p = activation[current]
                for succ in self.graph.successors(current):
                    w = self.graph[current][succ].get("weight", 1.0)
                    new_p = cur_p * w
                    if succ not in activation or new_p > activation[succ]:
                        activation[succ] = new_p
                        queue.append(succ)
            self.graph.add_edges_from(in_edges)   # 恢复
        activation.pop(cause, None)
        return activation

    def counterfactual_probs(self, observed_cause: str, observed_effect: str,
                             alternative_cause: str) -> Optional[float]:
        path = self.find_path(observed_cause, observed_effect)
        if not path:
            return None
        with self._lock:
            out_edges = list(self.graph.out_edges(observed_cause, data=True))
            self.graph.remove_edges_from(out_edges)
            activation = self.intervene(alternative_cause)
            self.graph.add_edges_from(out_edges)
        return activation.get(observed_effect, 0.0)
