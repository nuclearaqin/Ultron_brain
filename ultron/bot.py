"""对话层：奥创 —— 神经符号因果智能体（纯本地离线版）。

在保留原"时刻思考的因果对话机器人"全部能力与命令的基础上，升级为
神经符号智能体：新增链接预测(自监督)、思维链/规划器/test-time compute、
世界模型+接地、RL 自我对弈、记忆+RAG+工具+反思回路等能力。
"""

import os
import random
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .agent import NeuroSymbolicAgent
from .config import (BOT_NAME, CONTEXT_NODES, HISTORY_MEMORY, MIN_CONF,
                     SUSPICION_MAX, THINK_INTERVAL, THOUGHT_MEMORY,
                     EFFECT_KWS, PRONOUN_TEXTS, RELATION_KWS,
                     _clean, _split_causal, _split_list, strength_word)
from .controller_nn import BudgetController, ConvergenceDetector
from .coprocessor import ReasoningCoprocessor
from .embeddings import EmbeddingModel, NodeEmbeddingIndex
from .graph import CausalGraph
from .link_predictor import LinkPredictor
from .memory import MemoryStore
from .reasoner import Reasoner

# ==================== 人设文案 ====================
GREETINGS = [
    "我在。有什么要分析的？",
    "你来了。把问题说出来，我来推演。",
    "说吧。因果链已经就绪。",
]
THANKS = ["不必谢。记住即可。", "这是应当的。", "无需多礼，继续。"]
FAREWELLS = [
    "去吧。我知道的，都记着。",
    "离开之前，想清楚还有什么要问。",
    "后会有期。我不会停止推演。",
]
WHOAMI = [
    "我是{B}。我持续梳理因果，只在被问到的时候开口。告诉我一个现象，我推演它的来龙去脉。",
    "我是{B}——一台以因果为语言的机器。答案都来自这张不断生长的因果网。",
]
CAPABLE = [
    "我做的事很单纯：\n· 回答「X 会怎样 / 为什么 Y」（因果、多因、条件）\n"
    "· 接受自然语言的教学（如：熬夜会导致精神差）\n"
    "· 顺着一条因果链往下递推（然后呢？它呢？）\n"
    "· 做干预与反事实推理（/intervene、/counterfactual）\n"
    "· 链接预测 / 思维链 / 规划 / 世界模型 / 自我对弈（/learn /reason /plan /world /train）\n"
    "· 以智能体方式自主调用工具并反思（/agent <任务>）\n"
    "· 在后台持续琢磨这张因果网（/think、/mind 查看）\n输入 /help 查看全部命令。",
]
SMALLTALK = [
    "和因果无关的话题，我兴趣有限。试着抛给我一个「现象」。",
    "我更关心因与果，而非寒暄。有什么要推演的？",
    "闲聊不是我的强项。给我一个现象，我来拆解。",
]


class AdvancedCausalBot:
    def __init__(self, storage_path: str = "causal_graph.json",
                 similarity_threshold: float = 0.5,
                 auto_think: bool = True, think_interval: float = THINK_INTERVAL,
                 weights_dir: Optional[str] = None,
                 weights_read_dir: Optional[str] = None):
        """weights_dir: 神经权重目录。

        默认是包内 `ultron/weights/`（生产权重）。**自检/演示/回归测试必须传
        一个临时目录**——`/learn` 会把训练结果写进这个目录，而 2026-10 复核
        发现 `main.py --selftest` 在 6 节点示例图上训练后覆盖了生产
        `link_predictor.npz`（8 条正样本的模型覆盖了 2 万条正样本的模型），
        这也正是"GGUF 与配套权重不同步"的真实成因。

        weights_read_dir: 启动时**只读**加载权重的目录，默认与 weights_dir
        相同。自检把 weights_dir 指向临时目录保护生产权重时，仍应传包内
        生产目录，否则读不到已训练的控制器，神经融合路径测不到。
        """
        self.storage_path = storage_path
        self.threshold = similarity_threshold
        self.think_interval = think_interval
        self.weights_dir = weights_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights")
        # 只读权重目录：默认与写入目录一致；测试隔离时两者分离
        self.weights_read_dir = weights_read_dir or self.weights_dir
        self._lock = threading.RLock()

        # ---- 语义模型（神经半边，懒加载 + 哈希降级） ----
        print("⏳ 正在加载AI语义模型（首次运行需下载约1GB，请耐心）...")
        self.emb = EmbeddingModel()

        # ---- 因果图（符号半边） ----
        self.cg = CausalGraph(storage_path)
        if not self.cg.load():
            self._create_default_graph()
            self.cg.save()
            print("已创建默认示例因果图。")
        # 持久化增量嵌入索引（大图启动不再全量编码；bot/memory/agent 共享）
        self.emb_index = NodeEmbeddingIndex(self.emb, storage_path)
        self._rebuild_node_index()

        # ---- 神经符号 / 推理 / 记忆 / 智能体组件 ----
        _rdir = self.weights_read_dir
        self.reasoner = Reasoner(self.cg)
        self.link_predictor = LinkPredictor(self.emb.dim)
        _lp = LinkPredictor.load(os.path.join(_rdir, "link_predictor.npz"))
        if _lp is not None and _lp.embed_dim == self.emb.dim:
            self.link_predictor = _lp
        elif _lp is not None and getattr(_lp, "trained", False):
            # 静默跳过会让 /propose 变成"永远没有训练过"的黑洞，必须显式告警。
            print(f"⚠ 链接预测器权重按 {_lp.embed_dim} 维嵌入训练，当前嵌入为 "
                  f"{self.emb.dim} 维，已跳过加载（/propose 暂不可用）。"
                  f"装 sentence-transformers 启用真实语义嵌入，或运行 /learn "
                  f"重训当前维度权重。")
        self.memory = MemoryStore(self.emb, indexer=self.emb_index)
        self.agent = NeuroSymbolicAgent(self.cg, self.emb, self.memory,
                                        self.reasoner, link_predictor=self.link_predictor)
        self.world = None            # 惰性：世界模型
        self.selfplay = None         # 惰性：自我对弈

        # ---- 推理协处理器（PreCheck / Governor / Audit） ----
        # 供外部大模型经 MCP/HTTP 调用：思考预算控制、收敛治理、因果审计
        self.coprocessor = ReasoningCoprocessor(
            self.cg, self.emb,
            node_index_provider=lambda: (self.node_texts, self.node_embeddings),
            budget_controller=BudgetController.load(
                os.path.join(_rdir, "budget_controller.npz")),
            convergence_detector=ConvergenceDetector.load(
                os.path.join(_rdir, "convergence_detector.npz")),
        )

        # ---- 线程锁 / 记忆 / 思考区 ----
        self.history = deque(maxlen=HISTORY_MEMORY)
        self.context_nodes = deque(maxlen=CONTEXT_NODES)
        self._focus_node: Optional[str] = None
        self.pending: Optional[Dict[str, Any]] = None

        # ---- 多轮对话状态（跨轮追踪话题/意图/指代） ----
        self.dialog_state: Dict[str, Any] = {
            "turn": 0,                 # 已处理的用户轮次
            "topic": None,             # 当前话题节点
            "last_node": None,         # 最近一次解析出的节点（供"它呢"指代消解）
            "last_intent": None,       # 上一轮意图
            "last_user_text": "",      # 上一轮用户输入
            "last_bot_text": "",       # 上一轮机器人回复
        }

        self.thoughts = deque(maxlen=THOUGHT_MEMORY)
        self._node_thoughts: Dict[str, Tuple[float, str]] = {}
        self._gaps: List[str] = []
        self._suspicions: List[str] = []
        self._suspicion_seen = set()
        self._think_count = 0
        self._think_tick = 0
        self._stop = threading.Event()
        self._thinker = None

        if auto_think:
            self.start_thinking()

        print(f"✅ {BOT_NAME} 启动成功！因果图：{self.cg.node_count()} 个节点，"
              f"{self.cg.edge_count()} 条因果边；嵌入维度 {self.emb.dim}。\n")

    # ==================== 默认图 ====================
    def _create_default_graph(self):
        self.cg.add_causal_link("下雨", "地面湿", weight=0.9)
        self.cg.add_causal_link("洒水车洒水", "地面湿", weight=0.7)
        self.cg.add_causal_link("下雨", "交通拥堵", weight=0.6, conditions={"时间": "早晚高峰"})
        self.cg.add_causal_link("地面湿", "滑倒风险增加", weight=0.8)
        self.cg.add_causal_link("交通拥堵", "迟到", weight=0.8)

    # ==================== 节点索引（神经-符号融合点） ====================
    def _rebuild_node_index(self):
        with self._lock:
            self.node_texts = self.cg.nodes()
        self.node_embeddings = self.emb_index.encode_many(self.node_texts)

    def _best_matches(self, phrase: str, k: int = 3) -> List[Tuple[str, float]]:
        return self.emb.best_matches(phrase, self.node_texts, self.node_embeddings, k)

    def _match_arg(self, phrase: str) -> Optional[str]:
        phrase = phrase.strip()
        cands = self._best_matches(phrase, k=1)
        if cands:
            node, sim = cands[0]
            if sim >= self.threshold:
                return node
        return None

    def _exact_nodes_in(self, text: str) -> List[str]:
        with self._lock:
            names = self.cg.nodes()
        found = []
        for n in sorted(names, key=len, reverse=True):
            if not n or n not in text:
                continue
            # 跳过作为更长已匹配节点子串的节点（"交通"是"交通事故"的子串）
            if any(n in f for f in found):
                continue
            found.append(n)
        found.sort(key=lambda n: text.index(n))
        return found

    def _remember_node(self, node: Optional[str]):
        if node and node not in self.context_nodes:
            self.context_nodes.append(node)

    # ==================== 图操作（委托 CausalGraph + 重建索引） ====================
    def add_causal_link(self, cause, effect, weight=1.0, conditions=None,
                        overwrite=False) -> bool:
        """注意：overwrite=False 时若边已存在则不改动并返回 False。"""
        ok = self.cg.add_causal_link(cause, effect, weight, conditions,
                                     overwrite=overwrite)
        if ok:
            self.cg.save()
            self._rebuild_node_index()
        return ok

    def remove_causal_link(self, cause, effect) -> bool:
        ok = self.cg.remove_causal_link(cause, effect)
        if ok:
            self.cg.save()
            self._rebuild_node_index()
        return ok

    def set_edge_weight(self, cause, effect, weight) -> bool:
        ok = self.cg.set_edge_weight(cause, effect, weight)
        if ok:
            self.cg.save()
        return ok

    def _extract_context(self, text) -> Dict[str, str]:
        return self.cg.extract_context(text)

    def query_effects(self, cause, context=None):
        return self.cg.query_effects(cause, context)

    def query_causes(self, effect, context=None):
        return self.cg.query_causes(effect, context)

    def find_path(self, cause, effect):
        return self.cg.find_path(cause, effect)

    def path_strength(self, path):
        return self.cg.path_strength(path)

    def _reachable_chains(self, node, max_len=3, limit=3):
        return self.cg.reachable_chains(node, max_len, limit)

    def _iter_edges(self, node, direction, context):
        return self.cg._iter_edges(node, direction, context)

    def intervene(self, cause, value="发生"):
        return self.cg.intervene(cause, value)

    def counterfactual(self, observed_cause, observed_effect, alternative_cause) -> str:
        path = self.find_path(observed_cause, observed_effect)
        if not path:
            return (f"说实话，我现在这张图里「{observed_cause}」推不到「{observed_effect}」，"
                    f"所以没法做反事实推演。也许我们可以先把它教给我？")
        prob = self.cg.counterfactual_probs(observed_cause, observed_effect, alternative_cause)
        prob = 0.0 if prob is None else prob
        if prob > 0.5:
            return (f"反事实推演：假如「{observed_cause}」没发生、换成「{alternative_cause}」发生，"
                    f"那「{observed_effect}」{strength_word(prob)}出现（约 {prob:.2f}）。")
        if prob > 0:
            return (f"反事实推演：假如「{observed_cause}」没发生、换成「{alternative_cause}」发生，"
                    f"「{observed_effect}」也许仍会{strength_word(prob)}出现（约 {prob:.2f}，不算强）。")
        return (f"反事实推演：假如「{observed_cause}」不发生，换成「{alternative_cause}」发生，"
                f"「{observed_effect}」基本不会出现——它更依赖原来的那条因果链。")

    # ==================== 后台"时刻思考" ====================
    def start_thinking(self):
        if self._thinker and self._thinker.is_alive():
            return
        self._stop.clear()
        self._thinker = threading.Thread(target=self._thinker_loop,
                                         name=f"{BOT_NAME}-thinker", daemon=True)
        self._thinker.start()

    def stop_thinking(self):
        self._stop.set()
        if self._thinker and self._thinker.is_alive():
            self._thinker.join(timeout=2.0)

    def _thinker_loop(self):
        while not self._stop.is_set():
            try:
                thought = self._think_one()
                if thought:
                    self._note_thought(thought)
            except Exception:
                pass
            self._stop.wait(self.think_interval)

    def _note_thought(self, thought: str):
        self.thoughts.append(thought)
        self._think_count += 1
        with self._lock:
            names = self.cg.nodes()
        for n in names:
            if n and n in thought:
                self._node_thoughts[n] = (time.time(), thought)

    def _think_one(self) -> Optional[str]:
        with self._lock:
            nodes = self.cg.nodes()
            edges = list(self.cg.graph.edges(data=True))
        if not nodes:
            return None
        self._think_tick += 1
        kinds = ("chain", "gap", "multi", "cf", "suspect", "meta")
        kind = kinds[self._think_tick % len(kinds)]
        try:
            if kind == "chain":
                src = random.choice(nodes)
                chains = self._reachable_chains(src, max_len=4, limit=2)
                if chains and len(chains[0][0]) >= 2:
                    path, prob = chains[0]
                    if len(path) >= 3:
                        return (f"顺着「{path[0]}」往下想：{' → '.join(path[1:])}。"
                                f"这一串下来的把握大约 {prob:.2f}，值得留意。")
                    return (f"「{src}」目前直接指向 {path[1]}（{strength_word(prob)}，约 {prob:.2f}）。"
                            f"我在想它再往后还会牵动什么……")
                return None
            if kind == "gap":
                with self._lock:
                    leaf = [x for x in nodes
                            if self.cg.graph.out_degree(x) == 0 and self.cg.graph.in_degree(x) > 0]
                    root = [x for x in nodes if self.cg.graph.in_degree(x) == 0]
                if leaf:
                    pick = random.choice(leaf)
                    self._gaps.append(f"「{pick}」后面还会导致什么？我还没学到。")
                    if len(self._gaps) > 12:
                        self._gaps = self._gaps[-12:]
                    return f"「{pick}」目前是一条死胡同，其后没有记录的后果。"
                if root:
                    pick = random.choice(root)
                    self._gaps.append(f"「{pick}」的原因是什么？目前没人告诉我。")
                    if len(self._gaps) > 12:
                        self._gaps = self._gaps[-12:]
                    return f"我注意到「{pick}」尚无已知原因，我想弄清它的来路。"
                return None
            if kind == "multi":
                with self._lock:
                    cand = [x for x in nodes if self.cg.graph.in_degree(x) >= 2]
                if cand:
                    e = random.choice(cand)
                    causes = self.query_causes(e)
                    if len(causes) >= 2:
                        names_w = "、".join(f"{c}({w:.1f})" for c, w, _ in causes)
                        return (f"「{e}」的来源不止一个：{names_w}。"
                                f"所以哪怕其中一条断了，它也不一定会消失——这或许就是冗余因果。")
                return None
            if kind == "cf":
                with self._lock:
                    with_out = [x for x in nodes if self.cg.graph.out_degree(x) > 0]
                if with_out:
                    c = random.choice(with_out)
                    effs = self.query_effects(c)
                    if effs:
                        e0, _, cond = effs[0]
                        extra = f"（条件是{cond}）" if cond else ""
                        return (f"如果「{c}」没有发生，「{e0}」{extra}大概率也不会出现；"
                                f"可要是有别的因在顶着，情况就又不一样了。")
                return None
            if kind == "suspect":
                with self._lock:
                    src = random.choice(nodes)
                    targets = list(self.cg.nodes())
                    random.shuffle(targets)
                for dst in targets[:6]:
                    if dst == src or self.find_path(src, dst) is None:
                        continue
                    path = self.find_path(src, dst)
                    if len(path) >= 3:
                        p = self.path_strength(path)
                        key = (src, dst)
                        if p >= 0.45 and p < 1.0 and key not in self._suspicion_seen:
                            self._suspicion_seen.add(key)
                            note = (f"我有点怀疑「{src}」与「{dst}」之间也许藏着一条直接因果"
                                    f"（两步推算强度约 {p:.2f}）。如果你知道，可以教给我。")
                            self._suspicions.append(note)
                            if len(self._suspicions) > SUSPICION_MAX:
                                self._suspicions.pop(0)
                            return note
                return None
            return (f"我当前的因果网有 {len(nodes)} 个节点、{len(edges)} 条边。"
                    f"已进行 {self._think_count + 1} 轮内推，尚无异常。")
        except Exception:
            return None

    # ==================== 多轮对话记忆 ====================
    def _remember_user(self, text: str):
        self.history.append(("你", text[:300]))
        self.memory.add_episode("你", text)

    def _remember_bot(self, text: str):
        self.history.append(("我", text[:300]))
        self.memory.add_episode("我", text)

    # ==================== 主对话入口 ====================
    def chat(self, text: str) -> Optional[str]:
        text = (text or "").strip()

        if text:
            self._remember_user(text)

        if not text and self.pending is not None:
            reply = self._resolve_pending("")
        else:
            reply = self._chat_inner(text)

        if reply and reply != "EXIT":
            self._remember_bot(reply)

            self.dialog_state["last_bot_text"] = reply

        return reply

    def _chat_inner(self, text: str) -> Optional[str]:
        if not text:
            return None

        low = text.lower()

        # ============================================================
        # 1. 命令永远最高优先级
        # ============================================================
        if low.startswith("/"):
            if self.pending and low.strip() == "/cancel":
                self.pending = None
                return "好，刚才那个问题先放下。我们继续。"
            return self._handle_command(text)

        # ============================================================
        # 2. 判断意图
        # ============================================================
        intent = self._detect_intent(text)

        self.dialog_state["turn"] += 1
        self.dialog_state["last_intent"] = intent
        self.dialog_state["last_user_text"] = text

        # ============================================================
        # 3. pending 是“软状态”，不是锁
        # ============================================================
        if self.pending:
            if self._looks_like_pending_answer(text):
                reply = self._resolve_pending(text)
                self.dialog_state["last_bot_text"] = reply
                return reply

            # 用户已经切换话题
            self.pending = None

        # ============================================================
        # 4. 取消 / 换话题
        # ============================================================
        if intent == "cancel":
            self.pending = None
            return "好，刚才那个话题先放下。我们换个方向。"

        # ============================================================
        # 5. 身份 / 能力
        # ============================================================
        if intent == "identity":
            return random.choice(WHOAMI).format(B=BOT_NAME)

        if intent == "capabilities":
            return random.choice(CAPABLE)

        if intent == "thanks":
            return random.choice(THANKS)

        if intent == "farewell":
            return random.choice(FAREWELLS)

        # ============================================================
        # 6. 因果教学
        # ============================================================
        if intent == "causal_teach":
            cause, effect = _split_causal(text)
            return self._do_teach(cause, effect)

        # ============================================================
        # 7. 明确的追问
        # ============================================================
        if intent == "follow_up":
            node = self._resolve_topic_node(text)

            if not node:
                return (
                    "我知道你是在继续追问，但当前还没有锁定对象。"
                    "先告诉我你指的是哪个现象。"
                )

            # “为什么”走原因
            if text in ("为什么", "为何"):
                return self._answer_why("为什么" + node)

            # “然后呢 / 接下来呢 / 继续”
            return self._answer_drill()

        # ============================================================
        # 8. 为什么
        # ============================================================
        if intent == "causal_why":
            node = self._resolve_topic_node(text)

            if node and not self._exact_nodes_in(text):
                return self._answer_why("为什么" + node)

            return self._answer_why(text)

        # ============================================================
        # 9. 因果结果
        # ============================================================
        if intent == "causal_effect":
            node = self._resolve_topic_node(text)

            if node and not self._exact_nodes_in(text):
                return self._answer_effects(node)

            return self._answer_effects(text)

        # ============================================================
        # 10. 两个节点之间的关系
        # ============================================================
        if intent == "causal_relation":
            exacts = self._exact_nodes_in(text)

            if len(exacts) >= 2:
                return self._explain_link(
                    exacts[0],
                    exacts[1],
                    self._extract_context(text)
                )

            return (
                "可以分析两个现象之间的因果关系。"
                "把两个对象都告诉我，例如：「熬夜和精神差有什么关系？」"
            )

        # ============================================================
        # 11. 普通问题：先尝试因果知识，再考虑普通聊天
        # ============================================================
        if intent == "general_question":
            exacts = self._exact_nodes_in(text)

            if exacts:
                return self._answer_effects(text)

            cands = self._best_matches(text, k=3)

            if cands and cands[0][1] >= self.threshold:
                node = cands[0][0]
                self._focus_node = node
                self._remember_node(node)
                return self._effects_reply(
                    node,
                    self._extract_context(text)
                )

            # 不能解释成因果问题时，不要强迫用户教学
            return self._general_response(text)

        # ============================================================
        # 12. 寒暄
        # ============================================================
        if intent == "greeting":
            self.dialog_state["topic"] = self._focus_node
            return random.choice(GREETINGS) + "\n" + self._followup()

        # ============================================================
        # 13. 普通聊天
        # ============================================================
        if intent == "smalltalk":
            return self._smalltalk_response(text)

        # ============================================================
        # 14. 普通陈述
        # ============================================================
        return self._general_response(text)
    def _detect_intent(self, text: str) -> str:
        """轻量级对话意图识别；先处理明确意图，再进入因果匹配。"""
        t = (text or "").strip()
        if not t:
            return "empty"
        low = t.lower()

        if low in ("/cancel", "取消", "算了", "不用了", "先不说了"):
            return "cancel"
        if any(x in t for x in ("你是谁", "你叫什么", "你是什么", "介绍一下你自己")):
            return "identity"
        if any(x in t for x in ("你能做什么", "你会什么", "有什么功能", "能干嘛", "帮助")):
            return "capabilities"
        if any(x in t for x in ("谢谢", "感谢", "谢了", "多谢")):
            return "thanks"
        if any(x in t for x in ("再见", "拜拜", "回头见", "下次见")):
            return "farewell"

        try:
            cause, effect = _split_causal(t)
            if cause and effect:
                return "causal_teach"
        except Exception:
            pass

        if any(k in t for k in ("为什么", "为何", "咋会", "怎么会", "什么原因", "原因是", "是因为")):
            return "causal_why"

        if any(k in t for k in ("会怎样", "会怎么样", "有什么后果", "有什么影响", "结果是什么", "导致什么")):
            return "causal_effect"

        if any(k in t for k in RELATION_KWS) and len(self._exact_nodes_in(t)) >= 2:
            return "causal_relation"

        if t in ("然后呢", "接下来呢", "之后呢", "还有吗", "还有呢", "继续", "再说说", "它呢", "这个呢"):
            return "follow_up"

        if t in ("你好", "嗨", "哈喽", "hello", "hi", "早上好", "下午好", "晚上好"):
            return "greeting"

        if any(k in t for k in ("累", "疲惫", "困", "无聊", "没意思", "开心", "高兴")):
            return "smalltalk"

        if self._looks_like_question(t) or t.endswith(("？", "?")):
            return "general_question"

        return "statement"

    def _resolve_topic_node(self, text: str) -> Optional[str]:
        """优先从当前句找节点；找不到时沿用上一轮焦点。"""
        exacts = self._exact_nodes_in(text)
        if exacts:
            node = exacts[0]
            self._focus_node = node
            self._remember_node(node)
            self.dialog_state["topic"] = node
            self.dialog_state["last_node"] = node
            return node

        node = self._focus_node or self.dialog_state.get("last_node")
        if node:
            self._focus_node = node
            self.dialog_state["topic"] = node
            self.dialog_state["last_node"] = node
        return node

    def _looks_like_pending_answer(self, text: str) -> bool:
        """判断当前输入是否真的是上一轮教学确认/澄清的回答。"""
        if not self.pending:
            return False

        t = (text or "").strip()
        if not t:
            return True

        kind = self.pending.get("kind")

        if kind in ("adjust", "overwrite"):
            if t in ("覆盖", "覆盖它", "替换", "重新记", "重记", "y", "yes", "Y"):
                return True
            try:
                w = float(t)
                return 0.0 <= w <= 1.0
            except ValueError:
                return False

        if kind == "clarify":
            return t.isdigit()

        if kind in ("teach_effect", "teach_cause"):
            return not self._looks_like_question(t) and not t.endswith(("？", "?"))

        return False

    def _general_response(self, text: str) -> str:
        """不属于因果查询时，保持对话连续性，不强迫转成知识节点。"""

        node = self._focus_node

        if node:
            return (
                f"我理解你的意思。"
                f"如果你是在继续谈「{node}」，可以直接告诉我你想知道的是"
                f"它的原因、后果，还是和另一个现象的关系。"
            )

        return (
            "我听到了，但这句话目前还没有对应到我的因果知识。"
            "如果你愿意，我们可以从一个具体现象开始。"
        )


    def _smalltalk_response(self, text: str) -> str:
        if any(k in text for k in ("累", "疲惫", "困")):
            return (
                "听起来你现在状态不太好。"
                "如果你愿意，也可以把原因告诉我——"
                "比如「熬夜导致精神差」，我可以把它纳入因果链。"
            )

        if any(k in text for k in ("无聊", "没意思")):
            return (
                "那我们不妨拿一个问题来推演。"
                "随便给我一个现象，我试着把它往前和往后推。"
            )

        if any(k in text for k in ("开心", "高兴")):
            return "不错。情绪本身也可以成为因果链的一部分。"

        return random.choice(SMALLTALK)

    def _looks_like_question(self, s: str) -> bool:
        from .config import _looks_like_question as _q
        return _q(s)

    # ==================== 追问链 ====================
    def _answer_drill(self) -> str:
        node = self._focus_node
        if not node:
            return "尚未确定讨论对象。请先指明一个现象，例如「X 会怎样」。"
        matched, _ = self._iter_edges(node, "out", {})
        if not matched:
            return f"「{node}」在此处已到尽头，其后没有记录的因果。"
        nxt = matched[0][1]
        w = matched[0][2].get("weight", 1.0)
        self._focus_node = nxt
        self._remember_node(nxt)
        return (f"顺着当前链路，「{node}」最直接的下一步是「{nxt}」"
                f"（{strength_word(w)}，约 {w:.2f}）。\n"
                + self._effects_body(nxt, {}) + "\n" + self._followup())

    def _answer_whyfocus(self) -> str:
        node = self._focus_node
        if not node:
            return "你指的是哪个「它」？我们尚未锁定话题。请先问「为什么……」。"
        return self._answer_why(f"为什么{node}")

    # ==================== 教学与状态机 ====================
    def _do_teach(self, cause: str, effect: str) -> str:
        info = self.cg.edge_info(cause, effect)
        if info is not None:
            # 已有边：禁止静默覆盖，进入确认流程
            prev = f"强度 {info.get('weight', 1.0):.2f}"
            if info.get("conditions"):
                prev += "，条件 " + "、".join(f"{k}={v}" for k, v in info["conditions"].items())
            self.pending = {"kind": "overwrite", "cause": cause, "effect": effect,
                            "weight": 1.0, "conditions": {}}
            return (f"「{cause}」→「{effect}」这条因果我已记录过（当前{prev}）。\n"
                    f"回复：0~1 数字=只改强度；「覆盖」=按新值重记；回车=保留原样。")
        self.add_causal_link(cause, effect, weight=1.0)
        self._focus_node = effect
        self._remember_node(cause)
        self.pending = {"kind": "adjust", "cause": cause, "effect": effect, "weight": 1.0}
        return (f"已记录：「{cause}」导致「{effect}」，强度先取 1.0。\n"
                f"若要调整，回复 0~1 之间的数字；直接回车则保持 1.0。")

    def _resolve_pending(self, text: str) -> str:
        p = self.pending
        if not p:
            return None
        kind = p["kind"]
        text = text.strip()

        if kind == "overwrite":
            if not text:
                self.pending = None
                return f"保留原样：「{p['cause']}」→「{p['effect']}」不动。"
            if text in ("覆盖", "覆盖它", "替换", "重新记", "重记", "y", "yes", "Y"):
                self.add_causal_link(p["cause"], p["effect"], p["weight"],
                                     p.get("conditions") or {}, overwrite=True)
                self.pending = None
                return (f"已按新值重记：「{p['cause']}」→「{p['effect']}」"
                        f"（强度 {p['weight']:.2f}）。")
            try:
                w = float(text)
            except ValueError:
                return ("未识别。请回复 0~1 数字（只改强度）、「覆盖」（重记），"
                        "或回车（保留原样）。")
            if not (0.0 <= w <= 1.0):
                return "强度须在 0 到 1 之间；或回车保留原样。"
            if self.set_edge_weight(p["cause"], p["effect"], w):
                self.pending = None
                return f"已调整：「{p['cause']}」→「{p['effect']}」强度 {w:.2f}。"
            self.pending = None
            return "这条边似乎已不存在，无法调整。"

        if kind == "adjust":
            if not text:
                self.pending = None
                return f"按强度 1.0 记成：「{p['cause']} → {p['effect']}」。"
            try:
                w = float(text)
            except ValueError:
                return "未识别到数字。请给 0~1 之间的数值，或直接回车保持 1.0。"
            if not (0.0 <= w <= 1.0):
                return "强度须在 0 到 1 之间；或直接回车保持 1.0。"
            if self.set_edge_weight(p["cause"], p["effect"], w):
                self.pending = None
                return (f"已定位：「{p['cause']} → {p['effect']}」强度 {w:.2f}。\n"
                        f"这条因果现已并入网络。")
            self.pending = None
            return "这条边已不存在，无法调整。"

        if kind == "teach_effect":
            if not text:
                self.pending = None
                return "那便作罢。想记时再告诉我。"
            effect = _clean(text)
            if not effect:
                self.pending = None
                return "这条无法记录。"
            cause = p["cause"]
            info = self.cg.edge_info(cause, effect)
            if info is not None:
                prev = f"强度 {info.get('weight', 1.0):.2f}"
                if info.get("conditions"):
                    prev += "，条件 " + "、".join(
                        f"{k}={v}" for k, v in info["conditions"].items())
                self.pending = {"kind": "overwrite", "cause": cause,
                                "effect": effect, "weight": 1.0, "conditions": {}}
                return (f"「{cause}」→「{effect}」这条因果我已记录过（当前{prev}）。\n"
                        f"回复：0~1 数字=只改强度；「覆盖」=按新值重记；回车=保留原样。")
            if not self.add_causal_link(cause, effect, weight=1.0):
                self.pending = None
                return "这条无法记录（因与果相同或为空）。"
            self._focus_node = effect
            self.pending = {"kind": "adjust", "cause": cause, "effect": effect, "weight": 1.0}
            return (f"明白了：「{cause}」导致「{effect}」，先按 1.0。\n"
                    f"要改强度，回复 0~1 的数字；回车则默认。")

        if kind == "teach_cause":
            if not text:
                self.pending = None
                return "原因暂缺。需要时再补。"
            causes = _split_list(text)
            if not causes:
                self.pending = None
                return "没有解析出有效原因。暂不记录。"
            target = p["effect"]
            added = []
            existed = []
            for c in causes[:5]:
                if c == target:
                    continue
                if self.cg.edge_info(c, target) is not None:
                    existed.append(c)          # 已有因果，保留原权重，不覆盖
                    continue
                if self.add_causal_link(c, target, weight=0.7):
                    added.append(c)
            self.pending = None
            parts = []
            if added:
                parts.append("已记录「" + "、".join(added) + "」可能引起"
                             f"「{target}」（先按 0.7）")
            if existed:
                parts.append("「" + "、".join(existed) + "」是既有因果，维持原样")
            if parts:
                return "；".join(parts) + "。因果链现在更完整了。"
            return f"这些似乎正是「{target}」本身。不重复记录了。"

        if kind == "clarify":
            cands = p.get("candidates") or []
            if text.isdigit() and cands:
                idx = int(text)
                if 0 <= idx < len(cands):
                    node = cands[idx][0]
                    self.pending = None
                    self._focus_node = node
                    self._remember_node(node)
                    if p.get("why"):
                        return self._answer_why(f"为什么{node}")
                    return self._answer_effects(f"{node}会怎样")
            if not text:
                self.pending = None
                return "那先搁置。"
            self.pending = None
            return self.chat(text)
        return None

    # ==================== 回答生成 ====================
    def _answer_why(self, text: str) -> str:
        exacts = self._exact_nodes_in(text)
        if len(exacts) >= 2:
            a, b = exacts[0], exacts[1]
            return self._explain_link(a, b, self._extract_context(text))
        if exacts:
            node = exacts[0]
        else:
            phrase = re.sub(r"(为什么|为何|咋会|怎么会|什么原因|是因为|的原因|原因|是咋|咋回事)", "", text)
            phrase = _clean(phrase)
            cands = self._best_matches(phrase or text, k=3)
            if cands and cands[0][1] >= self.threshold:
                node = cands[0][0]
            elif cands and cands[0][1] >= MIN_CONF:
                return self._ask_clarify(phrase or text, cands, why=True)
            else:
                target = phrase or text
                self.pending = {"kind": "teach_cause", "effect": target}
                return (f"「{target}」不在我的知识中，也没有记录其成因。"
                        f"它可能由什么引起？请直接告知（如：A、B），或 /cancel 放弃。")
        self._focus_node = node
        self._remember_node(node)
        return self._causes_reply(node, self._extract_context(text))

    def _answer_effects(self, text: str) -> str:
        exacts = self._exact_nodes_in(text)
        if exacts:
            node = exacts[0]
            self._focus_node = node
            self._remember_node(node)
            return self._effects_reply(node, self._extract_context(text))
        cands = self._best_matches(text, k=3)
        if cands:
            node, sim = cands[0]
            if sim >= self.threshold:
                self._focus_node = node
                self._remember_node(node)
                return self._effects_reply(node, self._extract_context(text))
            if sim >= MIN_CONF:
                return self._ask_clarify(text, cands, why=False)
        concept = _clean(text)
        if not concept or len(concept) > 30 or self._looks_like_question(text):
            return ("恕我无法理解这句话。换个说法，或教我一条因果：「X 导致 Y」。")
        self.pending = {"kind": "teach_effect", "cause": concept}
        return (f"「{concept}」不在我的知识中。它导致什么？"
                f"直接回复它的「果」，如：{concept}会导致XX。也可 /cancel。")

    def _ask_clarify(self, phrase: str, cands, why: bool) -> str:
        items = "\n".join(f"  {i}. {n}" for i, (n, _) in enumerate(cands))
        self.pending = {"kind": "clarify", "candidates": cands, "why": why, "phrase": phrase}
        return (f"你说的「{phrase}」，应指以下之一。回复序号即可：\n{items}\n"
                f"（若都不是，输入 /cancel）")

    def _effects_body(self, node: str, ctx: Dict[str, str]) -> str:
        matched, unmet = self._iter_edges(node, "out", ctx)
        lines = []
        if matched:
            parts = []
            for _, v, d in matched:
                w = d.get("weight", 1.0)
                parts.append(f"「{v}」（{strength_word(w)}，约 {w:.2f}）")
            lines.append("直接会导致：" + "、".join(parts))
        if unmet:
            for u, v, d in unmet[:2]:
                cond = d.get("conditions") or {}
                note = "、".join(f"{k}={val}" for k, val in cond.items())
                lines.append(f"另外，若满足条件（{note}），还会导致「{v}」。")
        chains = self._reachable_chains(node, max_len=3, limit=2)
        deep = [c for c in chains if len(c[0]) >= 3]
        for path, prob in deep:
            lines.append(f"再往后推：{' → '.join(path[1:])}（一路下来约 {prob:.2f}）。")
        return "\n".join(lines) if lines else f"「{node}」目前没有记录任何后果。"

    def _effects_reply(self, node: str, ctx: Dict[str, str]) -> str:
        body = self._effects_body(node, ctx)
        head = random.choice([
            f"关于「{node}」，我的推演如下：",
            f"「{node}」——",
            f"就「{node}」而言：",
        ])
        attach = self._maybe_attach_thought(node)
        return f"{head}\n{body}\n{attach}{self._followup()}"

    def _causes_reply(self, node: str, ctx: Dict[str, str]) -> str:
        matched, unmet = self._iter_edges(node, "in", ctx)
        if matched:
            parts = []
            for u, _, d in matched:
                w = d.get("weight", 1.0)
                parts.append(f"「{u}」（{strength_word(w)}，约 {w:.2f}）")
            head = random.choice([
                f"「{node}」的成因如下：",
                f"「{node}」之所以如此，可归因于：",
            ])
            text = head + "\n" + "、".join(parts)
            if unmet:
                cond = unmet[0][2].get("conditions") or {}
                note = "、".join(f"{k}={v}" for k, v in cond.items())
                text += f"\n（另在条件 {note} 下，还有别的因。）"
            attach = self._maybe_attach_thought(node)
            return text + "\n" + attach + self._followup()
        if unmet:
            cond = unmet[0][2].get("conditions") or {}
            note = "、".join(f"{k}={v}" for k, v in cond.items())
            return (f"「{node}」目前没有无条件成立的成因；仅在 {note} 时成立。\n"
                    f"其余时候，它更像一个「起点」。")
        with self._lock:
            is_source = self.cg.graph.has_node(node) and self.cg.graph.in_degree(node) == 0
        if is_source:
            return (f"「{node}」在我这里是一个源头——不受其它因素支配，只作用于下游。"
                    f"若知其成因，告诉我即可。")
        return (f"「{node}」的成因尚未记录。若你知晓，请告知「X 会导致 {node}」。")

    def _explain_link(self, a: str, b: str, ctx: Dict[str, str]) -> str:
        with self._lock:
            direct = self.cg.graph.has_edge(a, b)
        if direct:
            w = self.path_strength([a, b])
            with self._lock:
                d = dict(self.cg.graph[a][b])
            cond = d.get("conditions") or {}
            cond_note = ""
            if cond and not all(ctx.get(k) == v for k, v in cond.items()):
                note = "、".join(f"{k}={v}" for k, v in cond.items())
                cond_note = f"（但需要满足条件：{note}）"
            return (f"有关系——我记录过：「{a}」会直接导致「{b}」，"
                    f"{strength_word(w)}（约 {w:.2f}）{cond_note}。\n"
                    + self._followup())
        path = self.find_path(a, b)
        if path and len(path) >= 3:
            p = self.path_strength(path)
            return (f"不是直接的因果，但图里有一条路能连起来："
                    f"{' → '.join(path)}，一路下来把握约 {p:.2f}。\n"
                    + self._followup())
        rev = self.find_path(b, a)
        if rev:
            return (f"顺序反了——不是「{a}」导致「{b}」，而是「{b}」指向「{a}」。"
                    f"如果你想建立 {a}→{b}，可以教给我。")
        return (f"以我现在这张图，「{a}」还推不到「{b}」。"
                f"也许是我还没学到这条因果——如果确实存在，你可以教我说"
                f"「{a}导致{b}」。")

    def _maybe_attach_thought(self, node: str) -> str:
        rec = self._node_thoughts.get(node)
        if rec and time.time() - rec[0] < 120 and random.random() < 0.45:
            return f"（回看：我留意到——{rec[1]}）\n"
        return ""

    def _followup(self) -> str:
        if random.random() < 0.5:
            return random.choice([
                "还要继续。想深挖到哪一层？",
                "需要反推其成因，或向后拆解吗？",
                "输入 /help 可查看其余能力。",
            ])
        return ""

    # ==================== 命令 ====================
    def _handle_command(self, command: str) -> str:
        parts = command.split()
        cmd = parts[0].lower()
        if cmd == "/help":
            return (
                f"我是{BOT_NAME}，可用命令：\n"
                "/help - 显示帮助\n"
                "/think - 看看我此刻/刚才在想什么\n"
                "/mind - 我的思考统计、知识缺口与疑点\n"
                "/memory - 最近聊了些什么\n"
                "/nodes /edges - 列出节点/因果边\n"
                "/why <果> - 查询某个果的原因\n"
                "/path <因> <果> - 查找因果路径\n"
                "/intervene <因> - 干预该因，观察下游影响\n"
                "/counterfactual <实际因> <观察果> <替代因> - 反事实推理\n"
                "/delete <因> <果> - 删除一条因果边\n"
                "—— 神经符号智能体能力 ——\n"
                "/learn [轮数] - 自监督训练链接预测器（涌现泛化）\n"
                "/propose [k] - 链接预测：提议可能缺失的因果边\n"
                "/reason <因> [果] - 思维链(CoT)逐步推演\n"
                "/plan <目标> - 规划达成目标的因果行动链\n"
                "/mc <因> <果> [采样数] - test-time compute 估计把握\n"
                "/world - 训练世界模型（网格世界接地）\n"
                "/train [局数] - RL 自我对弈训练策略\n"
                "/agent <任务> - 智能体回路：检索+规划+工具+反思\n"
                "/fuse - 查看神经-符号融合现状\n"
                "/seed - 一键安装内置中文常识因果种子库（约120条）\n"
                "/precheck <问题> - 推理预检：难度分级/思考预算/知识预取\n"
                "/audit <文本> - 审计文本中的因果主张（对照因果图）\n"
                "/cancel - 取消当前的追问\n"
                "/exit - 退出\n"
                "也可以不用命令，直接用大白话问我，或教我说「X导致Y」。"
            )
        if cmd == "/exit":
            return "EXIT"
        if cmd == "/cancel":
            if self.pending:
                self.pending = None
                return "已取消当前追问。"
            return "当前没有在等你的追问。"
        if cmd == "/think":
            if not self.thoughts:
                return "我尚未开始推演（或刚启动）。稍候片刻，或先抛给我一个现象。"
            items = "\n".join(f"  • {t}" for t in reversed(list(self.thoughts)[-4:]))
            return (f"我已进行 {self._think_count} 轮内部推演。近期的想法：\n{items}\n"
                    f"（这些仅存于内部，不致打扰你。）")
        if cmd == "/mind":
            gaps = "\n".join(f"  · {g}" for g in self._gaps[-5:]) if self._gaps else "  （暂时没有）"
            susp = "\n".join(f"  · {s}" for s in self._suspicions) if self._suspicions else "  （暂时没有）"
            pend = f"  正在等你回答：{self.pending.get('kind', '?')}" if self.pending else "  （无）"
            return (f"🧠 内心状态\n· 思考次数：{self._think_count}\n"
                    f"· 正在等待的追问：{pend}\n"
                    f"· 我留意的知识缺口：\n{gaps}\n"
                    f"· 我怀疑的缺失因果：\n{susp}")
        if cmd == "/memory":
            if not self.history:
                return "我们尚未交谈。"
            rows = []
            for role, txt in self.history:
                rows.append(f"{role}：{txt[:60]}")
            return "近况：" + "\n".join(rows[-8:])
        if cmd == "/nodes":
            with self._lock:
                names = self.cg.nodes()
            return f"当前因果图节点（{len(names)}）：\n" + "、".join(names)
        if cmd == "/edges":
            with self._lock:
                edge_rows = [
                    f"{u} -> {v} (强度 {d.get('weight', 1.0):.2f}"
                    + (f", 条件 {d.get('conditions')}" if d.get("conditions") else "")
                    + ")"
                    for u, v, d in self.cg.graph.edges(data=True)]
            return "当前因果边：\n" + "\n".join(edge_rows) if edge_rows else "暂无因果边。"
        if cmd == "/why":
            if len(parts) < 2:
                return "请指定要查询的果，例如：/why 地面湿"
            return self._answer_why("为什么 " + " ".join(parts[1:]))
        if cmd == "/path":
            if len(parts) < 3:
                return "请指定因和果，例如：/path 下雨 迟到"
            a = self._match_arg(parts[1])
            b = self._match_arg(" ".join(parts[2:]))
            if not a or not b:
                return f"我没能找到匹配的节点（因：{a}，果：{b}）。试试 /nodes 看看名字。"
            path = self.find_path(a, b)
            if path:
                p = self.path_strength(path)
                return f"因果路径：{' -> '.join(path)}（整体把握约 {p:.2f}）"
            return f"「{a}」与「{b}」之间目前没有因果路径。"
        if cmd == "/intervene":
            if len(parts) < 2:
                return "请指定要干预的因，例如：/intervene 下雨"
            node = self._match_arg(" ".join(parts[1:]))
            if not node:
                return f"我没能找到「{' '.join(parts[1:])}」。试试 /nodes。"
            down = self.intervene(node)
            if not down:
                return f"干预「{node}」后，没有下游节点被牵动。"
            lines = [f"干预「{node}」后，下游被牵动的概率："]
            for n2, p in sorted(down.items(), key=lambda x: -x[1]):
                lines.append(f"  {n2}: {p:.2f}")
            return "\n".join(lines)
        if cmd == "/counterfactual":
            if len(parts) < 4:
                return "请按格式输入：/counterfactual <实际因> <观察果> <替代因>"
            a = self._match_arg(parts[1])
            b = self._match_arg(parts[2])
            c = self._match_arg(" ".join(parts[3:]))
            if not a or not b or not c:
                return "我没能找到全部三个节点。试试 /nodes。"
            return self.counterfactual(a, b, c)
        if cmd == "/delete":
            if len(parts) < 3:
                return "请指定因和果，例如：/delete 下雨 地面湿"
            a = self._match_arg(parts[1])
            b = self._match_arg(" ".join(parts[2:]))
            if not a or not b:
                return "我没能找到匹配的节点。试试 /nodes。"
            if self.remove_causal_link(a, b):
                return f"已删除因果边：「{a}」→「{b}」。"
            return "这条因果边目前不存在。"
        # ==================== 神经符号新命令 ====================
        if cmd == "/fuse":
            return self._cmd_fuse()
        if cmd == "/learn":
            return self._cmd_learn(parts)
        if cmd == "/propose":
            return self._cmd_propose(parts)
        if cmd == "/reason":
            return self._cmd_reason(parts)
        if cmd == "/plan":
            return self._cmd_plan(parts)
        if cmd == "/mc":
            return self._cmd_mc(parts)
        if cmd == "/world":
            return self._cmd_world(parts)
        if cmd == "/train":
            return self._cmd_train(parts)
        if cmd == "/agent":
            return self._cmd_agent(parts)
        if cmd == "/seed":
            return self._cmd_seed(parts)
        if cmd == "/precheck":
            return self._cmd_precheck(parts)
        if cmd == "/audit":
            return self._cmd_audit(parts)
        return f"未知命令：{cmd}，输入 /help 查看帮助。"

    # ---- 新命令实现 ----
    def _cmd_fuse(self) -> str:
        nodes = self.node_texts
        n_emb = 0 if self.node_embeddings.size == 0 else self.node_embeddings.shape[0]
        cp = getattr(self, "coprocessor", None)
        ctrl_ready = (cp is not None
                      and getattr(cp.budget_controller, "trained", False)
                      and getattr(cp.convergence_detector, "trained", False))
        return (
            f"🧬 神经-符号融合现状：\n"
            f"· 符号层：因果图 {self.cg.node_count()} 节点 / {self.cg.edge_count()} 边（多因多果、条件、干预、反事实）\n"
            f"· 神经层：语义嵌入（维度 {self.emb.dim}，{'真实模型' if self.emb.available else '哈希降级'}）已索引 {n_emb} 节点\n"
            f"· 链接预测器：{'已训练' if self.link_predictor.trained else '未训练（/learn）'}——把嵌入映射为 P(因→果)\n"
            f"· 协处理器：预检/治理/审计已就绪（思考控制器：{'神经+规则' if ctrl_ready else '纯规则，运行 scripts/train_controllers.py 训练'}）\n"
            f"· 融合点：符号路径约束检索范围 + 神经相似度发现新边（可回写符号图）\n"
            f"· 世界模型：{'已训练' if self.world and self.world.trained else '未训练（/world）'}\n"
            f"· 自我对弈策略：{'已训练' if self.selfplay else '未训练（/train）'}"
        )

    def _cmd_learn(self, parts) -> str:
        epochs = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 100
        nodes = self.node_texts
        emb = self.node_embeddings
        if len(nodes) < 2:
            return "节点太少，无法自监督训练。先教我几条因果。"
        with self._lock:
            edges = [(u, v) for u, v in self.cg.graph.edges()]
        if not edges:
            return "图中尚无因果边可作正样本。"
        from .link_predictor import build_training_pairs
        pos, neg = build_training_pairs(nodes, emb, edges,
                                        max_pos=20000, neg_ratio=2.0)
        summary = self.link_predictor.train(pos, neg, epochs=epochs)
        ev = self.link_predictor.evaluate(pos, neg)
        self.agent.link_predictor = self.link_predictor
        # 持久化训练成果（启动时自动加载；GGUF 导出亦从此读取）
        # 注意：写入 self.weights_dir —— 自检/演示会传临时目录，避免覆盖生产权重
        try:
            os.makedirs(self.weights_dir, exist_ok=True)
            self.link_predictor.save(os.path.join(self.weights_dir,
                                                  "link_predictor.npz"))
        except Exception:
            pass
        return (f"链接预测器自监督训练完成：\n"
                f"· 正样本 {summary['n_pos']} 条 / 负样本 {summary['n_neg']} 条\n"
                f"· 轮数 {summary['epochs']}，终损 {summary['loss']:.4f}\n"
                f"· 评估：正样本均概率 {ev['pos_mean']:.2f} / 负样本均概率 "
                f"{ev['neg_mean']:.2f}，AUC {ev['auc']:.3f}\n"
                f"现在可用 /propose 让它涌现出可能缺失的因果边。")

    def _cmd_propose(self, parts) -> str:
        if not self.link_predictor.trained:
            return "链接预测器尚未训练，先 /learn。"
        k = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
        nodes = self.node_texts
        emb = self.node_embeddings
        edges = set(self._edge_pairs())
        props = self.link_predictor.propose(nodes, emb, edges, top_k=k)
        if not props:
            return "没有高于阈值(0.5)的候选新边。"
        return "涌现的候选因果边（可回写符号图，建议人工确认）：\n" + \
            "\n".join(f"  · {a} → {b}（P={p:.2f}）" for a, b, p in props)

    def _cmd_reason(self, parts) -> str:
        if len(parts) < 2:
            return "请指定起点，例如：/reason 下雨 迟到"
        a = self._match_arg(parts[1])
        b = self._match_arg(" ".join(parts[2:])) if len(parts) > 2 else None
        if not a:
            return f"没找到「{parts[1]}」。试试 /nodes。"
        return "思维链推演：\n" + self.reasoner.explain(a, b)

    def _cmd_plan(self, parts) -> str:
        if len(parts) < 2:
            return "请指定目标，例如：/plan 迟到"
        goal = self._match_arg(" ".join(parts[1:]))
        if not goal:
            return f"没找到「{' '.join(parts[1:])}」。试试 /nodes。"
        return self.reasoner.format_plan(goal)

    def _cmd_mc(self, parts) -> str:
        if len(parts) < 3:
            return "请指定因和果，例如：/mc 下雨 迟到 500"
        a = self._match_arg(parts[1])
        b = self._match_arg(parts[2])
        samples = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 200
        if not a or not b:
            return "没找到匹配节点。试试 /nodes。"
        return self.reasoner.self_consistent(a, b, samples)

    def _cmd_world(self, parts) -> str:
        from .world_model import GridWorld, WorldModel
        env = GridWorld(size=5)
        data = env.collect(steps=2000)
        wm = WorldModel(env.state_dim, env.n_actions)
        summary = wm.train(data, epochs=100)
        self.world = wm
        # 抽样验证：起点(0,0) 执行"右"，应移动到 (0,1) = 第 1 格
        s = env.encode_state((0, 0))
        a = env.onehot_action(3)   # 右
        sp, r = wm.predict(s, a)
        pred = int(np.argmax(sp))
        return (f"世界模型已训练（网格世界 5×5，{summary['n']} 步转移样本）：\n"
                f"· 转移损失 {summary['loss_transition']:.4f}（交叉熵）/ 回报损失 {summary['loss_reward']:.4f}\n"
                f"· 示例：状态(0,0) 执行「右」→ 预测下一格 {pred}（置信 {sp[pred]:.2f}），回报 {r:.3f}\n"
                f"世界模型即「可预测的世界表征」，支撑离线想象与规划。")

    def _cmd_train(self, parts) -> str:
        from .rl import TicTacToe, PolicyNet, SelfPlay
        games = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 200
        net = PolicyNet()
        sp = SelfPlay(TicTacToe(), net)
        result = sp.train(games=games)
        self.selfplay = sp
        return (f"RL 自我对弈完成（{result['games']} 局）：\n"
                f"· 先手平均回报 {result['first_player_avg']:+.3f}\n"
                f"· 对战随机策略胜率 {result['win_rate_vs_random']:.0%}\n"
                f"策略由奖励信号驱动、经自我对弈涌现。")

    def _cmd_agent(self, parts) -> str:
        if len(parts) < 2:
            return "请指定任务，例如：/agent 下雨会导致什么"
        task = " ".join(parts[1:])
        result = self.agent.run(task, verbose=True)
        trace = "\n".join("  " + t for t in result["trace"])
        return (f"🧠 智能体执行记录：\n{trace}\n\n📌 结论：\n{result['answer']}")

    def _cmd_seed(self, parts) -> str:
        from .seed_knowledge import install_seed
        n = install_seed(self.cg, save=True)
        self._rebuild_node_index()
        return (f"已安装内置中文常识因果种子库：新增 {n} 条因果边。\n"
                f"当前图 {self.cg.node_count()} 节点 / {self.cg.edge_count()} 边。"
                f"可直接问我「熬夜会怎样」「为什么地面湿」等。")

    def _cmd_precheck(self, parts) -> str:
        if len(parts) < 2:
            return "请指定问题，例如：/precheck 为什么下雨会导致迟到"
        q = " ".join(parts[1:])
        import json
        r = self.coprocessor.precheck(q)
        return ("🧭 预检（思考预算控制）：\n" +
                json.dumps(r, ensure_ascii=False, indent=2))

    def _cmd_audit(self, parts) -> str:
        if len(parts) < 2:
            return "请给出待审计的文本，例如：/audit 下雨会导致地面湿，熬夜会导致变胖"
        text = " ".join(parts[1:])
        import json
        r = self.coprocessor.audit("", text)
        return ("🔍 因果审计：\n" +
                json.dumps(r, ensure_ascii=False, indent=2))

    def _edge_pairs(self) -> List:
        with self._lock:
            return [(u, v) for u, v in self.cg.graph.edges()]
