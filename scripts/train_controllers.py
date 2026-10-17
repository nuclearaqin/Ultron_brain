"""训练思考控制器（预算控制器 + 收敛检测器）并保存权重。

用法：
    python scripts/train_controllers.py [--per-class 1500] [--trajs 900]

产物：
    ultron/weights/budget_controller.npz      难度控制器（13维 -> 0/1/2）
    ultron/weights/convergence_detector.npz   收敛检测器（9维 -> P(stop)）
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultron.controller_nn import BudgetController, ConvergenceDetector   # noqa: E402
from ultron.coprocessor import ReasoningCoprocessor                      # noqa: E402
from ultron.embeddings import EmbeddingModel, NodeEmbeddingIndex         # noqa: E402
from ultron.graph import CausalGraph                                     # noqa: E402
from ultron.synth_data import (build_budget_dataset,                     # noqa: E402
                               build_convergence_dataset)

WEIGHTS_DIR = os.path.join(ROOT, "ultron", "weights")


def split(X, y, test_ratio: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(int(len(X) * test_ratio), 1)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    all_scores = np.concatenate([scores_pos, scores_neg])
    order = all_scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    rank_pos = ranks[:len(scores_pos)].sum()
    return float((rank_pos - len(scores_pos) * (len(scores_pos) + 1) / 2.0)
                 / (len(scores_pos) * len(scores_neg)))


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    per_class = 1500
    n_trajs = 900
    if "--per-class" in argv:
        per_class = int(argv[argv.index("--per-class") + 1])
    if "--trajs" in argv:
        n_trajs = int(argv[argv.index("--trajs") + 1])

    t0 = time.time()
    cg = CausalGraph(os.path.join(ROOT, "causal_graph.json"))
    cg.load()
    emb = EmbeddingModel()
    index = NodeEmbeddingIndex(emb, os.path.join(ROOT, "causal_graph.json"))
    texts = cg.nodes()
    embs = index.encode_many(texts)
    cp = ReasoningCoprocessor(cg, emb, lambda: (texts, embs))
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ==================== 难度控制器 ====================
    print("=" * 60)
    print("合成难度数据集（0 跳 / 1 跳 / 多跳 任务模拟）...")
    X, y, stat = build_budget_dataset(cp, per_class=per_class)
    print(f"  样本 {stat['total']}（各难度分布："
          f"{[(int(k), int(v)) for k, v in zip(*np.unique(y, return_counts=True))]}）")
    print(f"  规则与真实难度一致率：{stat['rule_agreement']:.1%}"
          f"（其余为控制器可纠偏的空间）")
    Xtr, ytr, Xte, yte = split(X, y)
    bc = BudgetController(seed=0)
    info = bc.train(Xtr, ytr, epochs=220, lr=0.05, batch_size=64)
    pred = bc.predict_proba(Xte).argmax(axis=-1)
    acc_te = float((pred == yte).mean())
    print(f"  [训练] loss={info['loss']:.4f} acc={info['acc']:.3f} "
          f"n={info['n']} | [测试] acc={acc_te:.3f}")
    bc.save(os.path.join(WEIGHTS_DIR, "budget_controller.npz"))
    print(f"  已保存 -> {os.path.join(WEIGHTS_DIR, 'budget_controller.npz')}")

    # ==================== 收敛检测器 ====================
    print("=" * 60)
    print("合成收敛数据集（必要段 / 冗余段 轨迹模拟）...")
    Xc, yc, statc = build_convergence_dataset(cp, n_trajs=n_trajs)
    print(f"  样本 {statc['total']}（轨迹 {statc['n_trajectories']} 条，"
          f"stop 正例率 {statc['pos_rate']:.1%}）")
    Xtr, ytr, Xte, yte = split(Xc, yc)
    cd = ConvergenceDetector(seed=0)
    info = cd.train(Xtr, ytr, epochs=220, lr=0.05, batch_size=64)
    scores = cd.predict_proba(Xte).flatten()
    acc_te = float(((scores >= 0.5).astype(float) == yte).mean())
    pos_scores, neg_scores = scores[yte == 1], scores[yte == 0]
    auc = _auc(pos_scores, neg_scores) if len(pos_scores) and len(neg_scores) else float("nan")
    print(f"  [训练] loss={info['loss']:.4f} acc={info['acc']:.3f} "
          f"n={info['n']} | [测试] acc={acc_te:.3f} AUC={auc:.3f}")
    cd.save(os.path.join(WEIGHTS_DIR, "convergence_detector.npz"))
    print(f"  已保存 -> {os.path.join(WEIGHTS_DIR, 'convergence_detector.npz')}")

    print("=" * 60)
    print(f"全部完成，总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
