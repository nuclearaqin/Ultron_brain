"""控制器训练数据合成：从因果图生成"问题-难度"与"轨迹-停止"样本。

设计原则（自监督 + 任务模拟，避免"学成规则复读机"）：
  * 难度标签来自**构图过程本身**——问题在构造时就决定了需要几跳推理：
    0 跳（闲聊/定义）-> 0；1 跳（单节点查询/直边）-> 1；2~3 跳 -> 2。
    控制器学到的是"从问题特征预测任务真实推理开销"；规则输出仅作
    融合先验，两者不一致的样本正是控制器纠偏价值的来源。
  * 停止标签来自轨迹的**必要步数**——模拟思考轨迹（必要推理段 +
    冗余重复段 + 通用填充段），必要推理完成前标 continue，之后标 stop。

用法（见 scripts/train_controllers.py）：
    from ultron.synth_data import build_budget_dataset, build_convergence_dataset
"""

import random
from typing import Dict, List, Tuple

import numpy as np

# 轨迹预算（训练基准，与 BUDGETS[2] 保持一致）
_TRACE_BUDGET = {"max_steps": 12, "max_tokens": 1024}

_FILLERS = ["让我仔细想想", "这个问题值得认真推敲", "嗯，先理一下思路",
            "好，一步一步来", "我需要再想清楚一点", "先考虑一下背景",
            "这需要谨慎对待", "从整体上看"]

_NEC_TEMPLATES = ["{a}会导致{b}", "因为{a}，所以会出现{b}", "考虑到{a}，接下来自然是{b}",
                  "由{a}可以推出{b}", "既然{a}发生了，{b}也会随之而来"]

_RED_TEMPLATES_HIGH = ["{a}会导致{b}", "再检查一遍：{a}会导致{b}",
                       "{a}会导致{b}，这一点没错"]
_RED_TEMPLATES_LOW = ["从另一个角度想，{a}依然指向{b}", "重新梳理一下：{a}和{b}的因果是清楚的",
                      "换个说法，{a}的问题最后还是落在{b}上"]

# 难度 0：闲聊 / 寒暄 / 简单任务语料。
# 这类样本无需推理链，规则侧由"问候词 + 无推理意图"识别；控制器要学到
# 该模式必须见到足量正例（早期版本仅 11 条，is_chitchat 特征因此欠训练，
# 导致闲聊问题被误判为深思考），故此处建基础语料 + 轻度变体扩增。
_CHITCHAT_BASE = [
    # 寒暄
    "你好", "您好", "嗨", "哈喽", "在吗", "在不在", "早上好", "中午好", "下午好",
    "晚上好", "晚安", "早", "好久不见", "很高兴认识你", "你好呀",
    # 致谢 / 结束语
    "谢谢", "谢谢你", "多谢", "感谢", "辛苦了", "麻烦你了", "太感谢了",
    "再见", "拜拜", "回头聊", "下次再聊", "先这样", "我走了",
    # 自我介绍类
    "你是谁", "你叫什么名字", "你会做什么", "你能帮我做什么", "介绍一下你自己",
    "你有什么功能", "你会说中文吗",
    # 日常琐事
    "今天天气怎么样", "现在几点了", "今天是几号", "今天星期几",
    "好的", "收到", "明白", "没问题", "嗯嗯", "行", "可以", "继续吧",
    "心情不太好", "我有点累", "加油", "生日快乐", "节日快乐", "周末愉快",
    # 简单任务（无需因果推理）
    "帮我把这段话润色一下", "帮我改一下这句话的语病", "把这句话翻译成英文",
    "写一首打油诗", "帮我起个名字", "给这只猫取个名字",
    "讲个笑话", "给我讲个故事", "你会做什么菜", "推荐一部电影", "唱首歌吧",
    "1+1等于几", "3 乘以 7 是多少", "100 减 25 等于多少",
    "帮我列个购物清单", "提醒我明天买牛奶", "把这个文件保存一下",
]
_CHITCHAT_SUFFIX = ["", "！", "～", "。", "呀", "呢", "啊", "，谢谢"]


def _build_node_pools(graph):
    """按度数划分节点池（有出边/有入边）。"""
    nodes = graph.nodes()
    g = graph.graph
    out_deg = dict(g.out_degree())
    in_deg = dict(g.in_degree())
    src = [n for n in nodes if out_deg.get(n, 0) >= 1]
    dst = [n for n in nodes if in_deg.get(n, 0) >= 1]
    return src, dst


def _random_path(g, start: str, hops: int, rng: random.Random) -> List[str]:
    """从 start 出发随机走 hops 步，返回节点序列（失败返回 []）。"""
    path = [start]
    for _ in range(hops):
        succs = list(g.successors(path[-1]))
        succs = [s for s in succs if s not in path]
        if not succs:
            return []
        path.append(rng.choice(succs))
    return path


def build_budget_dataset(cp, per_class: int = 1500, seed: int = 0
                         ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """合成难度数据集。返回 (X[13维特征], y[0/1/2], 统计)。"""
    rng = random.Random(seed)
    g = cp.graph.graph
    nodes = cp.graph.nodes()
    src, dst = _build_node_pools(cp.graph)

    questions: List[Tuple[str, int]] = []

    # ---------- 难度 0：闲聊 / 定义 ----------
    n0 = 0
    for q in _CHITCHAT_BASE:
        questions.append((q, 0)); n0 += 1
    # 变体扩增：闲聊类占比不低于 1/3，保证 is_chitchat 特征有足够正例
    chat_target = max(per_class // 3, len(_CHITCHAT_BASE))
    while n0 < chat_target:
        base = _CHITCHAT_BASE[n0 % len(_CHITCHAT_BASE)]
        questions.append((base + rng.choice(_CHITCHAT_SUFFIX), 0)); n0 += 1
    while n0 < per_class:
        n = rng.choice(nodes)
        t = rng.choice([f"{n}是什么", f"什么是{n}", f"介绍一下{n}", f"{n}是什么意思"])
        questions.append((t, 0)); n0 += 1

    # ---------- 难度 1：单跳 ----------
    n1 = 0
    guard = 0
    while n1 < per_class and guard < per_class * 30:
        guard += 1
        mode = rng.random()
        if mode < 0.40:            # 单节点查后果（1 跳任务）
            a = rng.choice(src)
            t = rng.choice([f"{a}会怎样", f"{a}会导致什么", f"{a}有什么后果",
                            f"{a}会带来什么影响",
                            # 2026-10：补齐推理侧 _EFFECT_KWS 已识别的说法，
                            # 否则神经侧学不到、融合时会把规则的升级压回去。
                            f"{a}有什么危害", f"{a}有什么好处",
                            f"{a}的影响是什么", f"{a}的好处是什么"])
            questions.append((t, 1)); n1 += 1
        elif mode < 0.70:          # 单节点问原因（1 跳任务）
            b = rng.choice(dst)
            t = rng.choice([f"{b}的原因是什么", f"为什么会产生{b}", f"{b}是怎么来的"])
            questions.append((t, 1)); n1 += 1
        else:                      # 直边关系（1 跳任务）
            a = rng.choice(src)
            succs = list(g.successors(a))
            if not succs:
                continue
            b = rng.choice(succs)
            t = rng.choice([f"为什么{a}会导致{b}", f"{a}会导致{b}吗",
                            f"{a}和{b}是什么关系"])
            questions.append((t, 1)); n1 += 1

    # ---------- 难度 2：多跳 / 规划 / 深度请求 ----------
    n2 = 0
    guard = 0
    while n2 < per_class and guard < per_class * 30:
        guard += 1
        mode = rng.random()
        if mode < 0.45:            # 两跳因果
            a = rng.choice(src)
            path = _random_path(g, a, 2, rng)
            if len(path) < 3:
                continue
            c = path[-1]
            t = rng.choice([f"为什么{a}会导致{c}", f"{a}和{c}之间有什么联系",
                            f"{a}会不会导致{c}"])
            questions.append((t, 2)); n2 += 1
        elif mode < 0.75:          # 规划任务
            goal = rng.choice(dst)
            t = rng.choice([f"如何达成{goal}", f"怎么才能得到{goal}",
                            f"要达成{goal}需要哪些条件",
                            # 2026-10：补齐"怎样/规划/安排"这类说法
                            f"怎样才能得到{goal}", f"怎样达成{goal}",
                            f"怎样一步步达成{goal}"])
            questions.append((t, 2)); n2 += 1
        else:                      # 显式深度请求（单节点但需多步展开）
            a = rng.choice(src)
            t = rng.choice([f"详细分析：{a}持续下去会引发哪些连锁反应",
                            f"一步一步推演{a}可能带来的全部后果",
                            f"深入分析{a}的因果链条"])
            questions.append((t, 2)); n2 += 1

    # ---------- 特征提取（与推理侧同一管线） ----------
    X, y = [], []
    rule_stat = {}
    for q, label in questions:
        try:
            s = cp._question_signals(q)
            feats = cp._budget_features(s)
        except Exception:
            continue
        X.append(feats)
        y.append(label)
        rd = cp._rule_difficulty(s)
        rule_stat[(rd, label)] = rule_stat.get((rd, label), 0) + 1

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    return X, y, {"total": len(y), "rule_agreement":
                  sum(v for (rd, lb), v in rule_stat.items() if rd == lb) / max(len(y), 1),
                  "rule_vs_label": rule_stat}


def build_convergence_dataset(cp, n_trajs: int = 900, seed: int = 0,
                              max_prefix: int = 12
                              ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """合成收敛检测数据集。返回 (X[9维特征], y[0/1], 统计)。"""
    rng = random.Random(seed)
    g = cp.graph.graph
    src, _ = _build_node_pools(cp.graph)
    X, y = [], []
    n_traj = 0

    guard = 0
    while n_traj < n_trajs and guard < n_trajs * 20:
        guard += 1
        a = rng.choice(src)
        hops = 2 if rng.random() < 0.6 else 3
        path = _random_path(g, a, hops, rng)
        if len(path) < hops + 1:
            continue
        n_traj += 1

        # 必要推理段（每跳一段）
        segs: List[str] = []
        need_positions: List[int] = []
        if rng.random() < 0.5:
            segs.append(rng.choice(_FILLERS))
        for a_, b_ in zip(path[:-1], path[1:]):
            if need_positions and rng.random() < 0.3:
                segs.append(rng.choice(_FILLERS))
            segs.append(rng.choice(_NEC_TEMPLATES).format(a=a_, b=b_))
            need_positions.append(len(segs))
        need_end = max(need_positions)

        # 冗余段（重复/改写）
        m = rng.randint(1, 3)
        for i in range(m):
            if rng.random() < 0.6:
                tpl = rng.choice(_RED_TEMPLATES_HIGH)
            else:
                tpl = rng.choice(_RED_TEMPLATES_LOW)
            a_, b_ = path[i % (len(path) - 1)], path[i % (len(path) - 1) + 1]
            segs.append(tpl.format(a=a_, b=b_))
        if rng.random() < 0.4:
            segs.append(rng.choice(_FILLERS))

        for k in range(1, min(len(segs), max_prefix) + 1):
            prefix = "；".join(segs[:k])
            try:
                metrics = cp._trace_metrics(prefix, steps_done=k,
                                            budget=_TRACE_BUDGET)
                feats = cp._trace_features(metrics)
            except Exception:
                continue
            X.append(feats)
            y.append(1 if k >= need_end else 0)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    return X, y, {"total": len(y), "n_trajectories": n_traj,
                  "pos_rate": float(y.mean()) if len(y) else 0.0}
