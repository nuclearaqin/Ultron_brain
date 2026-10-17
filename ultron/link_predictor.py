"""链接预测器 —— 自监督学习 + 涌现式泛化 + 神经-符号融合的枢纽（目标 1/6）。

思路：把因果图当作"自监督标签"——已有的因果边是正样本，随机不存在的边是负样本。
用一个纯 numpy 的 MLP 学习 P(因→果)（输入为两节点语义嵌入的拼接），训练完成后：
  * 它能在没见过的新节点对之间预测"可能存在因果"，即"涌现式泛化 / 链接预测"；
  * 提议出的新边可回写符号因果图（经阈值 + 可选人工确认），完成神经→符号的闭环。
"""

import os
from typing import List, Tuple

import numpy as np

from .nn import MLP


def build_training_pairs(nodes: List[str], embeddings: np.ndarray, edges,
                         max_pos: int = 20000, neg_ratio: float = 2.0,
                         seed: int = 0) -> Tuple[List, List]:
    """构建链接预测训练集：正样本（已有边）+ 负样本（随机非边）。

    修复说明（v2）：旧实现在 bot._cmd_learn 里把负样本上限写成
    min(len(pos), n*4, 200)，大图上正负比高达 114951:200 ≈ 575:1，
    模型退化为"全部预测为正类"（终损 0.002、任何节点对 P≈1.00）。
    现改为：
      * 正样本下采样到 max_pos（11 万条边全量训练收益甚微且极慢）；
      * 负样本按 neg_ratio 倍正样本量随机采样，排除已有边；
      * 图越大随机对越不可能是边，直接采样即可（无需重试风暴）。
    """
    rng = np.random.default_rng(seed)
    idx = {n: i for i, n in enumerate(nodes)}
    pairs = [(u, v) for u, v in edges if u in idx and v in idx]
    if len(pairs) > max_pos:
        sel = rng.choice(len(pairs), size=max_pos, replace=False)
        pairs = [pairs[int(i)] for i in sel]
    pos = [(embeddings[idx[u]], embeddings[idx[v]]) for u, v in pairs]

    edge_set = set(edges)
    n = len(nodes)
    target_neg = int(len(pairs) * neg_ratio)
    neg = []
    attempts = 0
    max_attempts = target_neg * 50 + 1000
    while len(neg) < target_neg and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j or (nodes[i], nodes[j]) in edge_set:
            continue
        neg.append((embeddings[i], embeddings[j]))
    return pos, neg


class LinkPredictor:
    def __init__(self, embed_dim: int, hidden: Tuple[int, ...] = (256, 128), seed: int = 0):
        # 输入 = concat(因嵌入, 果嵌入)
        self.embed_dim = embed_dim
        self.net = MLP([2 * embed_dim, *hidden, 1],
                       hidden_act="tanh", out_act="sigmoid", seed=seed)
        self.trained = False

    def _feat(self, cause_emb: np.ndarray, effect_emb: np.ndarray) -> np.ndarray:
        return np.concatenate([cause_emb, effect_emb], axis=-1)

    def predict(self, cause_emb: np.ndarray, effect_emb: np.ndarray) -> float:
        x = self._feat(np.asarray(cause_emb).reshape(1, -1),
                       np.asarray(effect_emb).reshape(1, -1))
        return float(self.net.predict(x)[0, 0])

    def predict_batch(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        if not pairs:
            return np.array([])
        xs = np.stack([self._feat(c.reshape(-1), e.reshape(-1)) for c, e in pairs])
        return self.net.predict(xs).flatten()

    def train(self, pos_pairs: List[Tuple[np.ndarray, np.ndarray]],
              neg_pairs: List[Tuple[np.ndarray, np.ndarray]],
              epochs: int = 100, lr: float = 0.01, batch_size: int = 128,
              seed: int = 0) -> dict:
        """自监督训练：正边 label=1，负边 label=0。返回训练摘要。"""
        rng = np.random.default_rng(seed)
        X = [self._feat(c.reshape(-1), e.reshape(-1)) for c, e in pos_pairs + neg_pairs]
        y = np.array([1.0] * len(pos_pairs) + [0.0] * len(neg_pairs))
        X = np.array(X, dtype=float)
        if X.size == 0:
            return {"epochs": 0, "loss": float("nan"),
                    "n_pos": 0, "n_neg": 0}
        idx = np.arange(len(X))
        last = float("nan")
        for _ in range(epochs):
            rng.shuffle(idx)
            for b in range(0, len(idx), batch_size):
                batch = idx[b:b + batch_size]
                last = self.net.train_step(X[batch], y[batch], lr=lr, loss="bce")
        self.trained = True
        return {"epochs": epochs, "loss": last, "n_pos": len(pos_pairs),
                "n_neg": len(neg_pairs)}

    def evaluate(self, pos_pairs: List[Tuple[np.ndarray, np.ndarray]],
                 neg_pairs: List[Tuple[np.ndarray, np.ndarray]],
                 sample: int = 2000) -> dict:
        """评估：正/负样本的平均预测概率与 AUC（Mann-Whitney 秩统计）。

        AUC=0.5 表示与随机猜测无异（旧版 bug 后就是这个水平）；
        越接近 1 说明"存在因果边"与"不存在"的区分能力越强。
        """
        p = pos_pairs[:sample]
        q = neg_pairs[:sample]
        sp = self.predict_batch(p)
        sn = self.predict_batch(q)
        if sp.size == 0 or sn.size == 0:
            return {"pos_mean": float("nan"), "neg_mean": float("nan"),
                    "auc": float("nan")}
        all_scores = np.concatenate([sp, sn])
        order = all_scores.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(all_scores) + 1)
        rank_pos = ranks[:len(sp)].sum()
        auc = (rank_pos - len(sp) * (len(sp) + 1) / 2.0) / (len(sp) * len(sn))
        return {"pos_mean": float(sp.mean()), "neg_mean": float(sn.mean()),
                "auc": float(auc)}

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        arrays = {}
        for i, (w, b) in enumerate(zip(self.net.W, self.net.b)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str):
        """加载已训练的链接预测器；文件缺失或损坏返回 None。"""
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path)
            in_dim = int(data["W0"].shape[0])
            obj = cls(embed_dim=in_dim // 2)
            for i in range(len(obj.net.W)):
                obj.net.W[i] = np.asarray(data[f"W{i}"], dtype=float)
                obj.net.b[i] = np.asarray(data[f"b{i}"], dtype=float)
            obj.trained = True
            return obj
        except Exception:
            return None

    def propose(self, node_texts: List[str], node_embeddings: np.ndarray,
                existing_edges, top_k: int = 5,
                threshold: float = 0.5, max_candidates: int = 200_000,
                seed: int = 0) -> List[Tuple[str, str, float]]:
        """在无边的节点对里，按预测概率给出 top-k 条"可能缺失的因果边"。

        大图下 O(n²) 全量两两预测不可行（5 万节点约 28 亿对），故候选集改为：
          1) 图邻域推导：沿已有边补全"两步链"（u→v→w 建议 u→w；x→u→v 建议 x→v）；
          2) 随机采样非边对补齐候选规模；
        最后对候选矩阵做一次批量前向（向量化），秒级完成。
        """
        n = len(node_texts)
        existing = set(existing_edges)
        if n < 2 or max_candidates <= 0:
            return []
        idx = {t: i for i, t in enumerate(node_texts)}
        # ---- 候选集 1：两步链补全（比随机采样更有因果意义） ----
        succs = {}
        preds = {}
        for u, v in existing:
            succs.setdefault(u, []).append(v)
            preds.setdefault(v, []).append(u)
        pairs = set()
        for u, v in existing:
            for w in succs.get(v, [])[:10]:
                if w != u and (u, w) not in existing:
                    pairs.add((u, w))
            for x in preds.get(u, [])[:10]:
                if x != v and (x, v) not in existing:
                    pairs.add((x, v))
            if len(pairs) >= max_candidates:
                break
        # ---- 候选集 2：随机采样补齐（发现图结构之外的潜在因果） ----
        rng = np.random.default_rng(seed)
        trials = 0
        while len(pairs) < max_candidates and trials < max_candidates * 10:
            trials += 1
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i == j:
                continue
            a, b = node_texts[i], node_texts[j]
            if (a, b) not in existing:
                pairs.add((a, b))
        items = sorted(pairs)[:max_candidates]
        if not items:
            return []
        # ---- 批量前向（一次矩阵乘完成全部候选对打分） ----
        emb = np.asarray(node_embeddings, dtype=np.float32)
        X = np.stack([np.concatenate([emb[idx[a]], emb[idx[b]]]) for a, b in items])
        probs = self.net.predict(X).flatten()
        order = np.argsort(-probs)
        out: List[Tuple[str, str, float]] = []
        for k in order:
            if probs[k] < threshold:
                break
            a, b = items[k]
            out.append((a, b, float(probs[k])))
            if len(out) >= top_k:
                break
        return out
