"""全局常量与纯文本工具（从原 main.py 抽取，保持行为兼容）。"""

import re
from typing import List, Optional, Tuple

# ==================== 人设与常量 ====================
BOT_NAME = "奥创"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

THINK_INTERVAL = 2.5          # 后台思考节奏（秒）
THOUGHT_MEMORY = 24           # 保留最近多少条内心想法
HISTORY_MEMORY = 20           # 保留多少轮对话记忆
CONTEXT_NODES = 10            # 记住多少个近期提及的节点
MIN_CONF = 0.30               # 低于此相似度视为"没听懂"
SUSPICION_MAX = 8             # /mind 里最多保留几条"怀疑的缺失因果"

MAX_NODE_LEN = 40             # 节点文本长度上限（超过视为污染数据，加载时剔除）
MODEL_LOAD_TIMEOUT = 25.0     # 语义模型加载超时（秒）：超时则本会话降级为哈希特征

# ---- 口语化强度词 ----
def strength_word(w: float) -> str:
    if w >= 0.85:
        return "几乎一定会"
    if w >= 0.70:
        return "很可能会"
    if w >= 0.50:
        return "多半会"
    if w >= 0.30:
        return "有一定可能会"
    return "略微可能"


# ---- 意图关键词 ----
PRONOUN_TEXTS = {
    "它呢": "drill", "它": "drill", "这个": "drill", "那个": "drill",
    "然后呢": "drill", "然后": "drill", "接下来呢": "drill", "接下来": "drill",
    "那之后呢": "drill", "之后呢": "drill", "再往后呢": "drill", "再然后呢": "drill",
    "它的原因呢": "whyfocus", "它为什么": "whyfocus", "为什么会这样": "whyfocus",
    "为什么这样": "whyfocus", "这是为什么": "whyfocus", "那为什么": "whyfocus",
}
RELATION_KWS = ["有关系", "有关吗", "相关吗", "关联", "是不是", "能不能导致",
                "会影响", "会导致", "造成", "影响"]
EFFECT_KWS = ["会怎样", "会怎么样", "会咋样", "什么后果", "后果", "导致什么",
              "造成什么", "带来什么", "有什么影响", "会有什么影响", "影响",
              "接下来", "然后", "如果", "要是", "假如", "若"]
QWORDS = ["什么", "哪些", "怎样", "怎么样", "咋样", "如何", "为什么", "为何",
          "是不是", "会不会", "吗", "咋", "啥", "能不能"]
CAUSAL_VERBS = ["以至于", "从而导致", "进而导致", "导致", "造成", "引发", "引起",
                "带来", "使得", "致使", "促使", "触发", "加剧", "让", "使"]
EFFECT_LEAD = ["就会", "便会", "那么", "然后", "就", "则", "会", "将", "便",
               "让", "使", "人", "大家", "人们", "你", "我们", "别人"]
LEAD_NOISE = ["因为", "由于", "要是", "假如", "如果", "可能", "也许", "大概"]
TRAIL_NOISE = "了 的 呢 啊 吧 呀 哦 么 嘛 ！ ？ 。 ， 、 ； ： . ! ? , ; ~ ～"
TIME_ALIAS = {"早高峰": "早晚高峰", "晚高峰": "早晚高峰", "高峰期": "早晚高峰",
              "早上": "早上", "上午": "早上", "中午": "中午", "下午": "下午",
              "晚上": "晚上", "深夜": "深夜", "周末": "周末", "工作日": "工作日",
              "冬天": "冬天", "雨天": "雨天", "夏天": "夏天"}


# ==================== 纯文本工具 ====================
def _clean(s: str) -> str:
    return (s or "").strip().strip("，。！？；：、,.!?;: \t~～\"'‘’“”（）()[]【】")


def _strip_lead(s: str) -> str:
    s = _clean(s)
    changed = True
    while changed and s:
        changed = False
        for pre in LEAD_NOISE:
            if s.startswith(pre):
                s = _clean(s[len(pre):]); changed = True; break
        for pre in EFFECT_LEAD:
            if s.startswith(pre):
                s = _clean(s[len(pre):]); changed = True; break
    return s


def _strip_cause(s: str) -> str:
    """剥掉"因"尾部残留的助动词（如"熬夜会"->"熬夜"）。"""
    s = _clean(s)
    changed = True
    while changed and s:
        changed = False
        for suf in ("以至于", "可能会导致", "就会", "便会", "将会", "会",
                    "就", "则", "便", "将", "可能", "可以", "应该"):
            if s.endswith(suf):
                s = _clean(s[:-len(suf)])
                changed = True
                break
    return s


def _strip_trail(s: str) -> str:
    s = _clean(s)
    while s and s[-1] in TRAIL_NOISE.split():
        s = _clean(s[:-1])
    return s


def _looks_like_question(s: str) -> bool:
    return any(w in s for w in QWORDS) or s.endswith(("？", "?", "吗"))


def _split_causal(text: str) -> Optional[Tuple[str, str]]:
    """尝试把一句自然语言拆成 (因, 果)。拆不出返回 None。"""
    t = _clean(text)
    t = re.sub(r"^(教(我|你)?|帮我记(住|录)?|记(一下|住)?|添加一条|新增|加上)\s*[:：,，]?\s*", "", t)
    t = _clean(t)
    if not t or _looks_like_question(t):
        return None

    # 1) 箭头 / 因为…所以 / 如果…(就)…
    m = re.split(r"\s*(?:→|->|=>|⇒|⟹)\s*", t, maxsplit=1)
    if len(m) == 2:
        cause, effect = _strip_lead(m[0]), _strip_lead(m[1])
    else:
        m = re.match(r"^因为(.+?)所以(.+)$", t)
        if m:
            cause, effect = _strip_lead(m.group(1)), _strip_lead(m.group(2))
        else:
            m = re.match(r"^如果(.+?)(?:[，,]\s*|\s*(?:就|则|便会|就会|会|那么|那|便)\s*)(.+)$", t)
            if m:
                cause, effect = _strip_lead(m.group(1)), _strip_lead(m.group(2))
            else:
                # 2) 找第一个因果动词
                hits = []
                for v in CAUSAL_VERBS:
                    i = t.find(v)
                    if i >= 0:
                        hits.append((i, v))
                if not hits:
                    return None
                i, v = min(hits, key=lambda x: x[0])
                cause = _strip_trail(t[:i])
                effect = _strip_lead(t[i + len(v):])
                if not cause or not effect:
                    return None
    if not cause or not effect:
        return None
    cause, effect = _clean(_strip_cause(cause)), _clean(_strip_lead(_strip_trail(effect)))
    if not cause or not effect or effect == cause:
        return None
    if _looks_like_question(effect) or len(cause) > 24 or len(effect) > 24:
        return None
    return cause, effect


def _split_list(text: str) -> List[str]:
    """把用户回答里的一串原因拆成列表（A和B、A、B 等）。"""
    parts = re.split(r"[、，,;；/和与及以及跟]", text)
    out = []
    for p in parts:
        p = _strip_trail(_strip_lead(p))
        if p and p not in out:
            out.append(p)
    return out
