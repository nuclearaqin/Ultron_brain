"""思考控制器（神经组件 v2）：预算控制器 + 收敛检测器。

均为轻量 numpy MLP（复用 nn.py），训练数据由 synth_data.py 从因果图
合成。训练产物可通过 gguf_io.py 导出为 GGUF 权重容器。

设计约定（训练/推理两侧必须一致）：
  * BudgetController     输入 13 维问题特征 -> 难度分布 [p0, p1, p2]
  * ConvergenceDetector  输入  9 维轨迹特征 -> P(stop)
特征向量定义见 coprocessor.py 的 _budget_features / _trace_features。
特征维度变更后旧权重会被拒载（见 _ControllerBase.load），自动降级纯规则。
"""

import os
from typing import Optional, Tuple

import numpy as np

from .nn import MLP


class _ControllerBase:
    """公共训练/保存/加载逻辑。"""

    def __init__(self, dims, out_act, seed: int = 0):
        self.net = MLP(dims, hidden_act="tanh", out_act=out_act, seed=seed)
        self.trained = False
        self.train_info: dict = {}

    # ---------- 推理 ----------
    def predict_proba(self, X) -> np.ndarray:
        return self.net.predict(np.asarray(X, dtype=float))

    # ---------- 保存 / 加载 ----------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        arrays = {}
        for i, (w, b) in enumerate(zip(self.net.W, self.net.b)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> Optional["_ControllerBase"]:
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path)
            obj = cls(seed=0)
            for i in range(len(obj.net.W)):
                w = np.asarray(data[f"W{i}"], dtype=float)
                b = np.asarray(data[f"b{i}"], dtype=float)
                # 特征维度变更后旧权重形状不匹配：拒载并降级纯规则，
                # 避免"能加载但一推理就崩"的隐性故障。
                if w.shape != obj.net.W[i].shape or b.shape != obj.net.b[i].shape:
                    return None
                obj.net.W[i], obj.net.b[i] = w, b
            obj.trained = True
            return obj
        except Exception:
            return None

    # ---------- 通用小批量训练 ----------
    def _train_loop(self, X, Y, epochs: int, lr: float, batch_size: int,
                    loss: str, seed: int) -> float:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(X))
        last = float("nan")
        for _ in range(epochs):
            rng.shuffle(idx)
            for b in range(0, len(idx), batch_size):
                batch = idx[b:b + batch_size]
                last = self.net.train_step(X[batch], Y[batch], lr=lr, loss=loss)
        self.trained = True
        return last


class BudgetController(_ControllerBase):
    """难度控制器：问题特征(13维) -> 思考难度 0/1/2。"""

    def __init__(self, seed: int = 0):
        super().__init__([13, 32, 16, 3], out_act="softmax", seed=seed)

    def train(self, X: np.ndarray, y, epochs: int = 200, lr: float = 0.05,
              batch_size: int = 64, seed: int = 0) -> dict:
        """y: 整数标签 [N]（0/1/2）或 one-hot [N,3]。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if y.ndim == 1:
            Y = np.zeros((len(y), 3))
            Y[np.arange(len(y)), y.astype(int)] = 1.0
        else:
            Y = y.astype(float)
        loss = self._train_loop(X, Y, epochs, lr, batch_size, "ce", seed)
        pred = self.predict_proba(X).argmax(axis=-1)
        acc = float((pred == Y.argmax(axis=-1)).mean())
        self.train_info = {"epochs": epochs, "loss": loss, "acc": acc,
                           "n": int(len(X))}
        return self.train_info


class ConvergenceDetector(_ControllerBase):
    """收敛检测器：轨迹特征(9维) -> P(stop)。"""

    def __init__(self, seed: int = 0):
        super().__init__([9, 32, 16, 1], out_act="sigmoid", seed=seed)

    def train(self, X: np.ndarray, y, epochs: int = 200, lr: float = 0.05,
              batch_size: int = 64, seed: int = 0) -> dict:
        """y: 0/1 标签 [N]，1 = 应当停止。"""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(y, dtype=float).reshape(-1, 1)
        loss = self._train_loop(X, Y, epochs, lr, batch_size, "bce", seed)
        pred = (self.predict_proba(X) >= 0.5).astype(float)
        acc = float((pred == Y).mean())
        self.train_info = {"epochs": epochs, "loss": loss, "acc": acc,
                           "n": int(len(X))}
        return self.train_info
