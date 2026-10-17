"""纯 numpy 多层感知机 + 反向传播。

不依赖 torch/tensorflow，保证本地离线、零重依赖。供链接预测、世界模型、
强化学习策略网络等神经组件复用。
"""

from typing import List, Optional, Tuple

import numpy as np


def _init_weight(in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier 均匀初始化。"""
    s = np.sqrt(6.0 / (in_dim + out_dim))
    return rng.uniform(-s, s, (in_dim, out_dim))


class MLP:
    """通用前馈网络。

    激活函数：hidden_act ∈ {"tanh", "relu", "sigmoid"}（默认 tanh），
    out_act ∈ {None, "sigmoid", "softmax"}。softmax 配合交叉熵时由调用方
    直接给 logits 梯度，故此处不做额外处理。
    """

    def __init__(self, dims: List[int], hidden_act: str = "tanh",
                 out_act: Optional[str] = None, seed: int = 0):
        assert len(dims) >= 2, "dims 至少含输入/输出两层"
        rng = np.random.default_rng(seed)
        self.hidden_act = hidden_act
        self.out_act = out_act
        self.dims = list(dims)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i, o in zip(dims[:-1], dims[1:]):
            self.W.append(_init_weight(i, o, rng))
            self.b.append(np.zeros(o))
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]

    # ---------------- 激活 ----------------
    def _act(self, z: np.ndarray, name: Optional[str]) -> np.ndarray:
        if name == "tanh":
            return np.tanh(z)
        if name == "relu":
            return np.maximum(z, 0.0)
        if name == "sigmoid":
            z = np.clip(z, -30.0, 30.0)
            return 1.0 / (1.0 + np.exp(-z))
        if name == "softmax":
            z = z - np.max(z, axis=-1, keepdims=True)
            e = np.exp(np.clip(z, -30.0, 30.0))
            return e / e.sum(axis=-1, keepdims=True)
        return z

    def _d_hidden(self, a: np.ndarray) -> np.ndarray:
        if self.hidden_act == "tanh":
            return 1.0 - a * a
        if self.hidden_act == "relu":
            return (a > 0.0).astype(float)
        if self.hidden_act == "sigmoid":
            return a * (1.0 - a)
        return np.ones_like(a)

    # ---------------- 前向 ----------------
    def forward(self, x) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """返回 (输出, cache)。cache[l] = (a_l, z_{l+1})。"""
        x = np.asarray(x, dtype=float)
        a = x
        caches: List[Tuple[np.ndarray, np.ndarray]] = []
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            caches.append((a, z))
            a = self._act(z, self.hidden_act if l < L - 1 else self.out_act)
        return a, caches

    def predict(self, x) -> np.ndarray:
        out, _ = self.forward(x)
        return out

    # ---------------- 反向 ----------------
    def backward(self, d_out, caches, out) -> None:
        """把 d_out（对输出激活的梯度，或对 logits 的梯度）反传，累积到 dW/db。

        说明：out_act == "sigmoid" 时，d_out 视作对 sigmoid 输出的梯度，
        这里再乘 sigmoid' 得到对 logits 的梯度；softmax 的导数已由调用方
        在 d_out 中体现（dL/dlogits = pred - y）。
        """
        d = np.asarray(d_out, dtype=float)
        L = len(self.W)
        if self.out_act == "sigmoid":
            d = d * out * (1.0 - out)
        for l in reversed(range(L)):
            a_l, _ = caches[l]
            delta = d                     # 对 z_{l+1} 的梯度
            self.dW[l] += a_l.T @ delta
            self.db[l] += delta.sum(axis=0)
            if l > 0:
                d = delta @ self.W[l].T            # 对 a_l 的梯度
                d = d * self._d_hidden(a_l)        # 乘隐藏层激活导数 -> 对 z_l 的梯度

    def zero_grad(self) -> None:
        for i in range(len(self.W)):
            self.dW[i].fill(0.0)
            self.db[i].fill(0.0)

    def step(self, lr: float) -> None:
        for i in range(len(self.W)):
            self.W[i] -= lr * self.dW[i]
            self.b[i] -= lr * self.db[i]

    # ---------------- 训练一步 ----------------
    def train_step(self, x, y, lr: float = 0.01, loss: str = "mse") -> float:
        """单步训练。loss ∈ {"mse","bce","ce"}。返回标量损失。"""
        out, caches = self.forward(x)
        y = np.asarray(y, dtype=float).reshape(out.shape)
        n = x.shape[0] if hasattr(x, "shape") else 1
        if loss == "mse":
            d = 2.0 * (out - y) / n
            err = float(np.mean((out - y) ** 2))
        elif loss == "bce":
            # backward() 对 sigmoid 输出会再乘一次 p(1-p)（见 backward 说明），
            # 而此处 d 应是"对 logits 的梯度"(p-y)/n；先除以 p(1-p) 抵消，
            # 否则梯度被缩放 p(1-p)（中点 4 倍偏小、饱和区趋零），收敛显著变慢。
            denom = np.clip(out * (1.0 - out), 1e-9, None)
            d = (out - y) / (denom * n)
            p = np.clip(out, 1e-9, 1 - 1e-9)
            err = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        elif loss == "ce":
            d = (out - y) / n
            p = np.clip(out, 1e-9, 1.0)
            err = float(-np.mean(np.sum(y * np.log(p), axis=-1)))
        else:
            raise ValueError(f"未知损失：{loss}")
        self.zero_grad()
        self.backward(d, caches, out)
        self.step(lr)
        return err
