"""推理协处理器（Reasoning Co-Processor）：PreCheck / Governor / Audit。

定位：让大模型以低成本完成推理——Ultron 用符号因果图（毫秒级）承担
三类推理协调工作，替代 LLM 的长思维链开销：

  1. PreCheck（预检）：问题进入时——难度分级、思考预算建议、因果知识
     预取。LLM 拿到"值不值得深思 + 已知因果"再开工，避免"杀鸡用牛刀"。
  2. Governor（治理）：思考进行中——滚动检测收敛度/重复度/新颖度，
     判断"想够了没有"，建议提前截停，砍掉 overthinking 的冗余思考段。
  3. Audit（审计）：答案出口——抽取因果主张，逐条对照因果图校验，
     标注引用真伪，给出可信度评级。

三层均提供纯规则（v1）与神经增强（v2）双通道：预算控制器 / 收敛检测器
训练完成后自动融合；组件缺失时优雅降级为纯规则，功能完整可用。

所有返回值均为 JSON 友好结构，可直接经 MCP / HTTP 暴露给外部模型。
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import _strip_cause, _strip_lead

# ==================== 预算常量 ====================
DIFFICULTY_LABELS = {0: "免思考", 1: "轻思考", 2: "深思考"}
BUDGETS = {
    0: {"max_steps": 0, "max_tokens": 0},
    1: {"max_steps": 3, "max_tokens": 256},
    2: {"max_steps": 12, "max_tokens": 1024},
}

# ==================== 触发词表 ====================
_WHY_KWS = ("为什么", "为何", "咋会", "怎么会", "什么原因", "原因是", "是因为", "缘故",
            "怎么来的", "怎么来")
_EFFECT_KWS = ("会怎样", "会怎么样", "什么后果", "后果", "导致什么", "造成什么",
               "带来什么", "有什么影响", "会有什么影响", "接下来", "然后呢",
               "会导致", "会引发", "会引起", "会造成",
               # 2026-10 复核补充：这些同样常见的"问效果"说法原先漏网，
               # 于是"抽烟有什么危害""睡眠不足的影响是什么"被判免思考（预算 0）。
               "有什么危害", "有什么坏处", "有什么好处", "有什么作用",
               "有什么风险", "有什么效果", "有什么用", "有用吗",
               "有何影响", "有何危害", "影响是什么", "好处是什么",
               "危害是什么", "会有什么危害", "会有什么好处")
_PLAN_KWS = ("怎么办", "怎么才能", "怎样才能", "如何", "怎么达成", "计划", "方案",
             "办法", "规划", "安排", "步骤", "流程", "制定", "路线图", "怎么做到")
_FACT_KWS = ("是什么", "什么是", "介绍一下", "什么意思", "定义")
_RELATION_HINT = ("什么关系", "有什么关系", "什么联系", "有什么联系", "关联", "之间的关系")
_DEEP_KWS = ("详细分析", "深入分析", "全面分析", "一步步", "一步一步", "连锁反应",
             "仔细推演", "逐步推演", "完整推演")
_CHITCHAT_KWS = ("你好", "谢谢", "再见", "嗨", "哈喽", "hello", "hi", "在吗",
                 "早上好", "下午好", "晚上好")

# "怎样/怎么/咋" 类 how-to 请求的补丁（2026-10 复核发现）：
# _PLAN_KWS 原本只有"如何/怎么才能"，漏了"怎样"，导致
# "怎样规划一周的饮食？"这类**多步规划**问题被判成免思考（预算 0），
# 实测 40 题里 22 题给了 0 预算、其中多题实为规划题。
# 但不能只加"怎样"：''今天天气怎样？'' 是状态询问，不是规划。故要求
# 该疑问词后面**还有实质内容**，并排除"怎么回事/怎么样了"这类追问。
_HOW_KWS = ("怎样", "怎么", "咋")
_HOW_TAIL_STOP = ("回事", "了", "样", "办")   # 怎么回事 / 怎么样了 …


def _how_request(q: str) -> bool:
    """判断是否为 "怎样/怎么/咋 + 实质内容" 的 how-to（多步）请求。"""
    for k in _HOW_KWS:
        i = q.find(k)
        while i >= 0:
            tail = q[i + len(k):].strip("？?。！!，,、；;：: \t")
            if tail and not tail.startswith(_HOW_TAIL_STOP):
                return True
            i = q.find(k, i + 1)
    return False

# 规则-神经融合权重：语义确定性分支（闲聊/定义/显式深度请求）不依赖图匹配
# 与路径信息，规则结论可靠，给高权重；其余分支保留神经侧纠偏空间。
_RULE_CONF_STRONG = 0.90
_RULE_CONF_DEFAULT = 0.45

# 因果主张抽取：A + 因果动词 + B
_CAUSAL_VERBS_RE = re.compile(
    r"(导致|造成|引起|引发|带来|使得|会使|会让|将使|会让|致使|促使|引发|产生)"
)
_CLAIM_SPLIT_RE = re.compile(r"[，。；;！？!?\n]+")


def _jaccard(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram Jaccard 相似度（轻量、无依赖）。"""
    if not a or not b:
        return 0.0
    ga = {a[i:i + n] for i in range(max(len(a) - n + 1, 1))}
    gb = {b[i:i + n] for i in range(max(len(b) - n + 1, 1))}
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def _segments(text: str, min_len: int = 4) -> List[str]:
    """把思考文本切成段（句级）。"""
    parts = re.split(r"[。！？!?\n；;]+", text or "")
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


class ReasoningCoprocessor:
    """推理协处理器：PreCheck / Governor / Audit。"""

    def __init__(self, graph, embeddings, node_index_provider=None,
                 budget_controller=None, convergence_detector=None):
        """
        graph: CausalGraph 实例
        embeddings: EmbeddingModel 实例
        node_index_provider: 可选回调 () -> (node_texts, node_embeddings)，
            用于拿到最新的节点索引（图增长后自动生效）
        budget_controller / convergence_detector: 可选神经控制器
            （见 ultron/controller_nn.py），未训练时自动降级纯规则。
        """
        self.graph = graph
        self.emb = embeddings
        self._node_index_provider = node_index_provider
        self.budget_controller = budget_controller
        self.convergence_detector = convergence_detector
        self._node_set_cache: Optional[Tuple[int, set]] = None

    # ==================== 节点索引工具 ====================
    def _node_index(self) -> Tuple[List[str], np.ndarray]:
        if self._node_index_provider is None:
            return [], np.array([])
        try:
            return self._node_index_provider()
        except Exception:
            return [], np.array([])

    def _node_set(self, texts: List[str]) -> set:
        """按文本列表对象身份缓存节点集合（53k nodes 重建一次 ~5ms）。"""
        if self._node_set_cache is not None and self._node_set_cache[0] == id(texts):
            return self._node_set_cache[1]
        s = set(texts)
        self._node_set_cache = (id(texts), s)
        return s

    def _match_nodes_fast(self, text: str, texts: List[str],
                          max_node_len: int = 24) -> List[str]:
        """快速子串匹配：枚举 text 的 2~24 字子串查节点集合。

        相比逐节点 `n in text`（O(53k × len)），复杂度 O(len(text) × 24)，
        毫秒级完成；语义上与原 _exact_nodes_in 等价（均为子串匹配）。
        """
        node_set = self._node_set(texts)
        found: List[str] = []
        L = len(text)
        seen = set()
        for i in range(L):
            for j in range(i + 2, min(i + max_node_len, L) + 1):
                sub = text[i:j]
                if sub in node_set and sub not in seen:
                    seen.add(sub)
                    found.append(sub)
        # 去掉是更长命中节点子串的节点
        found = [n for n in found if not any(n != f and n in f for f in found)]
        found.sort(key=lambda n: text.index(n))
        return found

    # ==================== PreCheck ====================
    def _question_signals(self, question: str) -> Dict[str, Any]:
        """问题特征：图匹配 + 类型 + 路径信息（规则与神经共用）。"""
        texts, embs = self._node_index()
        q = (question or "").strip()
        exact = self._match_nodes_fast(q, texts) if texts else []

        sem_nodes: List[Tuple[str, float]] = []
        if not exact and texts and getattr(embs, "size", 0):
            try:
                sem_nodes = self.emb.best_matches(q, texts, embs, k=3)
            except Exception:
                sem_nodes = []

        # 类型判定
        is_why = any(k in q for k in _WHY_KWS)
        is_effect = any(k in q for k in _EFFECT_KWS)
        is_plan = any(k in q for k in _PLAN_KWS) or _how_request(q)
        is_deep = any(k in q for k in _DEEP_KWS)
        # "是什么关系/有什么联系"含"是什么/什么"字样，但本质是因果关系问题，
        # 不能误判为定义类（fact）；同理"介绍一下X会有什么影响"含定义词但
        # 推理意图才是主任务，故定义类要求**没有其它推理意图**同时出现。
        is_relation = any(k in q for k in _RELATION_HINT)
        is_fact = any(k in q for k in _FACT_KWS) \
            and not is_relation \
            and not (is_why or is_effect or is_plan or is_deep)
        # 闲聊：含问候词**且**没有其它推理意图（"你好，请详细分析…"是深度
        # 请求而非闲聊）；或极短输入
        is_chitchat = (any(k in q for k in _CHITCHAT_KWS)
                       and not (is_why or is_effect or is_plan or is_deep
                                or is_relation)
                       ) or len(q) <= 3

        # 图特征
        path_len = 0
        path_strength = 0.0
        path: List[str] = []
        if len(exact) >= 2:
            try:
                p = self.graph.find_path(exact[0], exact[1])
                if p:
                    path = p
                    path_len = len(p) - 1
                    path_strength = self.graph.path_strength(p)
            except Exception:
                pass
        degree = 0
        if exact:
            try:
                g = self.graph.graph
                degree = (g.out_degree(exact[0]) if g.has_node(exact[0]) else 0) + \
                         (g.in_degree(exact[0]) if g.has_node(exact[0]) else 0)
            except Exception:
                pass

        return {
            "question": q,
            "exact_nodes": exact[:4],
            "sem_nodes": sem_nodes,
            "n_exact": len(exact),
            "is_why": is_why, "is_effect": is_effect, "is_plan": is_plan,
            "is_fact": is_fact, "is_chitchat": is_chitchat, "is_deep": is_deep,
            "is_relation": is_relation,
            "path": path, "path_len": path_len, "path_strength": path_strength,
            "degree": degree, "q_len": len(q),
        }

    def _rule_difficulty(self, s: Dict[str, Any]) -> int:
        """规则难度：0 免思考 / 1 轻思考 / 2 深思考。"""
        if s["is_chitchat"]:
            return 0
        if s["is_fact"]:
            return 0   # 定义/介绍类问题不需要因果推理
        if s.get("is_deep"):
            return 2   # 显式要求深度推演：直接给满预算
        # 两节点关系/多跳 → 深思考
        if s["n_exact"] >= 2:
            if s["path_len"] >= 2 or s["is_plan"]:
                return 2
            if s["path_len"] == 1:
                return 1
            # 两节点但图中无路径：因果问句需探索性推理，否则只是句中
            # 恰好撞上两个节点名（如"今天天气怎么样"）→ 免思考
            if s["is_why"] or s["is_effect"] or s["is_relation"]:
                return 2
            return 0
        # 单节点 / 语义命中：**仅在带询问意图时**才算因果查询。
        # 注意语义匹配无阈值、恒返回 top-k，若只看"有无命中"会把
        # "推荐一部电影"这类恰好含节点名（电影）的闲聊也判为轻思考。
        if s["n_exact"] == 1 or s["sem_nodes"]:
            if s["is_plan"]:
                return 2
            if s["is_why"] or s["is_effect"]:
                return 1
            return 0
        # 无匹配：若带规划/多步意图 → 深思考；否则免思考
        if s["is_plan"]:
            return 2
        if s["is_why"] or s["is_effect"]:
            return 1
        return 0

    def precheck(self, question: str) -> Dict[str, Any]:
        """问题预检：难度分级 + 思考预算 + 因果知识预取。"""
        s = self._question_signals(question)
        rule_diff = self._rule_difficulty(s)
        difficulty = rule_diff
        controller_used = "rule"
        neural_probs: Optional[List[float]] = None

        if self.budget_controller is not None and getattr(self.budget_controller, "trained", False):
            try:
                feats = np.asarray([self._budget_features(s)], dtype=float)
                probs = self.budget_controller.predict_proba(feats)[0]  # [p0,p1,p2]
                prior = np.zeros(3)
                prior[rule_diff] = 1.0
                # 置信度加权融合：确定性分支上规则占主导（0.90），
                # 需要图路径判断的分支保留神经侧较大权重（0.45）。
                w = _RULE_CONF_STRONG if (s["is_chitchat"] or s["is_fact"]
                                          or s["is_deep"]) else _RULE_CONF_DEFAULT
                fused = (1.0 - w) * probs + w * prior
                difficulty = int(np.argmax(fused))
                controller_used = "rule+neural"
                neural_probs = [round(float(p), 3) for p in probs]
            except Exception:
                pass

        budget = dict(BUDGETS[difficulty])

        # 知识预取
        knowledge: List[Dict[str, Any]] = []
        for node in s["exact_nodes"][:2]:
            try:
                effs = self.graph.query_effects(node)[:6]
                causes = self.graph.query_causes(node)[:6]
            except Exception:
                effs, causes = [], []
            knowledge.append({
                "node": node,
                "effects": [[v, round(w, 3)] for v, w, _ in effs],
                "causes": [[u, round(w, 3)] for u, w, _ in causes],
            })

        result: Dict[str, Any] = {
            "question": s["question"],
            "difficulty": difficulty,
            "difficulty_label": DIFFICULTY_LABELS[difficulty],
            "budget": budget,
            "knowledge": knowledge,
            "controller": controller_used,
            "signals": {
                "matched_nodes": s["exact_nodes"],
                "semantic_matches": [[n, round(p, 3)] for n, p in s["sem_nodes"]],
                "path_len": s["path_len"],
                "path_strength": round(s["path_strength"], 3),
                "question_type": ("plan" if s["is_plan"] else "why" if s["is_why"]
                                  else "effect" if s["is_effect"] else "fact"
                                  if s["is_fact"] else "chat" if s["is_chitchat"]
                                  else "general"),
            },
        }
        if s["path"]:
            result["path"] = {
                "nodes": s["path"],
                "strength": round(s["path_strength"], 3),
            }
        if neural_probs is not None:
            result["neural_probs"] = neural_probs   # 融合前神经侧原始分布（可解释性）
        # 原因/后果问题：把直接关联一并给出，LLM 可直接引用
        if difficulty == 0 and not knowledge:
            result["guidance"] = "无需外部推理：直接回答即可（或本就无需因果知识）。"
        elif difficulty == 1:
            result["guidance"] = "轻量推演即可：优先引用预取因果，勿过度展开。"
        else:
            result["guidance"] = "需要多步推理：可沿预取路径展开，但每步务必与因果知识对齐。"
        return result

    @staticmethod
    def _budget_features(s: Dict[str, Any]) -> List[float]:
        """难度控制器特征向量（固定 13 维，训练/推理两侧严格一致）。"""
        return [
            min(s["n_exact"], 4) / 4.0,
            min(len(s["sem_nodes"]), 3) / 3.0,
            min(s["path_len"], 6) / 6.0,
            float(s["path_strength"]),
            min(s["degree"], 50) / 50.0,
            1.0 if s["is_why"] else 0.0,
            1.0 if s["is_effect"] else 0.0,
            1.0 if s["is_plan"] else 0.0,
            1.0 if s["is_fact"] else 0.0,
            1.0 if s["is_chitchat"] else 0.0,
            1.0 if s["is_deep"] else 0.0,
            min(s["q_len"], 100) / 100.0,
            1.0 if s["n_exact"] >= 2 else 0.0,
        ]

    # ==================== Governor ====================
    def _trace_metrics(self, thinking: str, steps_done: int,
                       budget: Optional[Dict]) -> Dict[str, float]:
        """思考轨迹指标：重复度 / 新颖度 / 图吻合度 / 预算消耗。"""
        segs = _segments(thinking)
        n = len(segs)
        total_len = len(thinking or "")
        metrics = {
            "segments": float(n),
            "length": float(total_len),
            "repetition_max": 0.0,
            "novelty": 1.0,
            "graph_hit_ratio": 0.0,
            "budget_ratio": 0.0,
        }
        if n == 0:
            return metrics

        # 重复度：最近 2 段与更早各段的最大 Jaccard（绕圈信号）
        recent = segs[-2:]
        earlier = segs[:-2]
        if earlier:
            rep = 0.0
            for r in recent:
                for e in earlier:
                    j = _jaccard(r, e)
                    if j > rep:
                        rep = j
            metrics["repetition_max"] = rep

        # 新颖度：最近 2 段中未在更早文本出现过的 2-gram 比例
        recent_text = "".join(recent)
        earlier_text = "".join(earlier)
        if recent_text:
            gr = {recent_text[i:i + 2] for i in range(max(len(recent_text) - 1, 1))}
            ge = {earlier_text[i:i + 2] for i in range(max(len(earlier_text) - 1, 1))}
            new_grams = gr - ge
            metrics["novelty"] = (len(new_grams) / len(gr)) if gr else 1.0

        # 图吻合度：句中出现的已知因果节点占比
        texts, _ = self._node_index()
        if texts:
            hit_segs = 0
            touched = set()
            for seg in segs[-6:]:
                nodes = self._match_nodes_fast(seg, texts)
                if nodes:
                    hit_segs += 1
                    touched.update(nodes)
            metrics["graph_hit_ratio"] = hit_segs / max(len(segs[-6:]), 1)
            metrics["known_nodes"] = float(len(touched))

        # 预算消耗
        if budget:
            max_tokens = max(int(budget.get("max_tokens", 0)), 1)
            max_steps = max(int(budget.get("max_steps", 0)), 1)
            ratio_t = total_len / (max_tokens * 2.2)   # 约 2.2 字符/token（中文粗估）
            ratio_s = max(steps_done, n) / max_steps
            metrics["budget_ratio"] = max(ratio_t, ratio_s)
        return metrics

    def _rule_stop(self, metrics: Dict[str, float]) -> Tuple[bool, float, str]:
        """规则判停：返回 (stop, confidence, reason)。偏保守——宁可多想一步。"""
        br = metrics.get("budget_ratio", 0.0)
        rep = metrics.get("repetition_max", 0.0)
        nov = metrics.get("novelty", 1.0)
        segs = metrics.get("segments", 0.0)

        if br >= 1.0:
            return True, 0.95, "预算耗尽：思考已达预设上限，应立即收束并作答。"
        if segs >= 4 and rep >= 0.85:
            return True, 0.85, "检测到明显重复：最近推理与此前高度重合，继续等价于原地绕圈。"
        if segs >= 6 and nov <= 0.08:
            return True, 0.75, "已收敛：最近数段几乎没有引入新信息，继续推理收益极低。"
        if segs >= 8 and metrics.get("graph_hit_ratio", 0.0) >= 0.5 and nov <= 0.2:
            return True, 0.65, "已贴近已知因果：推理脉络与因果知识高度吻合且新信息趋零。"
        return False, 0.5, "尚未收敛：仍在引入新信息，建议继续（受预算约束）。"

    def govern(self, question: str, thinking: str, steps_done: int = 0,
               budget: Optional[Dict] = None) -> Dict[str, Any]:
        """思考治理：检测收敛/重复/预算，给出停止或继续的建议。"""
        metrics = self._trace_metrics(thinking or "", steps_done, budget)
        stop, conf, reason = self._rule_stop(metrics)
        controller_used = "rule"

        if (not stop and self.convergence_detector is not None
                and getattr(self.convergence_detector, "trained", False)):
            try:
                feats = np.asarray([self._trace_features(metrics)], dtype=float)
                p_stop = float(self.convergence_detector.predict_proba(feats)[0])
                if p_stop >= 0.8 and metrics["segments"] >= 3:
                    stop = True
                    conf = max(0.6, p_stop * 0.85)
                    reason = f"收敛检测器判定：继续推理的边际价值低（P(stop)={p_stop:.2f}）。"
                    controller_used = "neural"
                elif p_stop <= 0.2:
                    controller_used = "rule+neural"
            except Exception:
                pass

        metrics_out = {k: round(v, 3) for k, v in metrics.items()}
        return {
            "stop": stop,
            "confidence": round(conf, 3),
            "reason": reason,
            "metrics": metrics_out,
            "controller": controller_used,
        }

    @staticmethod
    def _trace_features(m: Dict[str, float]) -> List[float]:
        """收敛检测器特征向量（固定 9 维，训练/推理两侧严格一致）。"""
        segs = m.get("segments", 0.0)
        return [
            min(segs, 24) / 24.0,
            min(m.get("length", 0.0), 4000) / 4000.0,
            m.get("repetition_max", 0.0),
            m.get("novelty", 1.0),
            m.get("graph_hit_ratio", 0.0),
            min(m.get("budget_ratio", 0.0), 2.0) / 2.0,
            1.0 if segs >= 4 else 0.0,
            1.0 if segs >= 8 else 0.0,
            min(m.get("known_nodes", 0.0), 20) / 20.0,
        ]

    # ==================== Audit ====================
    def _extract_claims(self, answer: str, max_claims: int = 12) -> List[Tuple[str, str, str]]:
        """从答案中抽取因果主张：(原句, 因, 果)。

        注意中文切分陷阱："下雨会导致地面湿" 的因果动词是 "导致"，
        但朴素切分会把 "下雨会" 当作因——故对因尾部剥助动词
        （_strip_cause），对果头部剥话头（_strip_lead）。
        """
        claims = []
        for sent in _CLAIM_SPLIT_RE.split(answer or ""):
            sent = sent.strip()
            if len(sent) < 6:
                continue
            m = _CAUSAL_VERBS_RE.search(sent)
            if not m:
                continue
            cause = _strip_cause(sent[:m.start()])
            effect = _strip_lead(sent[m.end():])
            if not cause or not effect or len(cause) > 24 or len(effect) > 24:
                continue
            if cause == effect:
                continue
            claims.append((sent, cause, effect))
            if len(claims) >= max_claims:
                break
        return claims

    def audit(self, question: str, answer: str) -> Dict[str, Any]:
        """答案审计：抽取因果主张并对照因果图校验一致性。"""
        texts, _ = self._node_index()
        node_set = self._node_set(texts) if texts else set()
        claims_out: List[Dict[str, Any]] = []
        issues: List[str] = []

        for sent, cause, effect in self._extract_claims(answer):
            c_known = cause in node_set
            e_known = effect in node_set
            direct = False
            path: Optional[List[str]] = None
            strength = 0.0
            try:
                direct = self.graph.has_edge(cause, effect)
                if direct:
                    strength = float(self.graph.path_strength([cause, effect]))
                else:
                    p = self.graph.find_path(cause, effect)
                    if p and len(p) >= 2:
                        path = p
                        strength = float(self.graph.path_strength(p))
            except Exception:
                pass

            if direct:
                verdict = "supported"
            elif path:
                verdict = "reachable"
            elif c_known and e_known:
                verdict = "unsupported"
            else:
                verdict = "unknown"

            claims_out.append({
                "claim": sent[:80],
                "cause": cause, "effect": effect,
                "cause_known": c_known, "effect_known": e_known,
                "direct": direct,
                "path": path,
                "strength": round(strength, 3),
                "verdict": verdict,
            })
            if verdict == "unsupported":
                issues.append(f"图中两节点存在但无法建立因果路径：{cause} → {effect}")
            elif verdict == "unknown":
                missing = [t for t, k in ((cause, c_known), (effect, e_known)) if not k]
                issues.append(f"引用了图中不存在的概念：{'、'.join(missing)}")

        if claims_out:
            score_map = {"supported": 1.0, "reachable": 0.6,
                         "unsupported": 0.25, "unknown": 0.1}
            score = sum(score_map[c["verdict"]] for c in claims_out) / len(claims_out)
        else:
            score = -1.0   # 无因果主张可审
            issues.append("未在该答案中检测到可校验的因果主张。")

        consistent = None if score < 0 else (score >= 0.5)
        return {
            "consistent": consistent,
            "score": round(score, 3) if score >= 0 else None,
            "n_claims": len(claims_out),
            "claims": claims_out,
            "issues": issues,
        }

    # ==================== 便捷入口 ====================
    def process(self, question: str, thinking: str = "", answer: str = "",
                steps_done: int = 0) -> Dict[str, Any]:
        """一站式：预检 + （有思考时）治理 + （有答案时）审计。"""
        out: Dict[str, Any] = {"precheck": self.precheck(question)}
        if thinking:
            out["govern"] = self.govern(
                question, thinking, steps_done=steps_done,
                budget=out["precheck"]["budget"])
        if answer:
            out["audit"] = self.audit(question, answer)
        return out
