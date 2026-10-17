"""世界模型 + 接地（目标 3）：可学习的环境转移模型 T(s,a)->s'。

"接地"意味着智能体的表征不再只是符号，而是与一个可交互的(物理)环境挂钩：
智能体在 GridWorld 里行动、观测、学习一个可预测的下一步状态与回报。
WorldModel 用纯 numpy MLP 拟合 T(s,a)->(s', r)，即"可预测的世界表征"。
"""

from typing import Dict, Tuple

import numpy as np

from .nn import MLP


class GridWorld:
    """简单网格世界：智能体从起点走到终点，撞墙不动，踩中目标得奖。"""

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 上/下/左/右
    ACTION_NAMES = ["上", "下", "左", "右"]

    def __init__(self, size: int = 5, goal: Tuple[int, int] = (4, 4),
                 start: Tuple[int, int] = (0, 0),
                 obstacles=None, seed: int = 0):
        self.size = size
        self.goal = goal
        self.start = start
        self.obstacles = set(obstacles or [])
        self.rng = np.random.default_rng(seed)
        self.pos = start
        self.n_actions = 4
        self.state_dim = size * size

    def reset(self) -> Tuple[int, int]:
        self.pos = self.start
        return self.pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        dy, dx = self.ACTIONS[action]
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        if 0 <= ny < self.size and 0 <= nx < self.size and (ny, nx) not in self.obstacles:
            self.pos = (ny, nx)
        done = (self.pos == self.goal)
        reward = 1.0 if done else -0.01
        return self.pos, reward, done

    def encode_state(self, pos: Tuple[int, int]) -> np.ndarray:
        v = np.zeros(self.state_dim, dtype=float)
        v[pos[0] * self.size + pos[1]] = 1.0
        return v

    def onehot_action(self, action: int) -> np.ndarray:
        v = np.zeros(self.n_actions, dtype=float)
        v[action] = 1.0
        return v

    def collect(self, steps: int = 2000) -> list:
        """随机策略采集 (s, a, s', r) 转移样本，用于训练世界模型。"""
        data = []
        self.reset()
        for _ in range(steps):
            s = self.pos
            a = int(self.rng.integers(0, self.n_actions))
            sp, r, done = self.step(a)
            data.append((self.encode_state(s), self.onehot_action(a),
                         self.encode_state(sp), r))
            if done:
                self.reset()
        return data


class WorldModel:
    """学习 T(s,a)->s' 与 R(s,a)->r 的神经网络世界模型。"""

    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, ...] = (128, 128),
                 seed: int = 0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 下一步状态 = 分类预测（softmax + 交叉熵）；tanh 在此稀疏 one-hot 任务上比 relu 收敛更好
        self.transition = MLP([state_dim + action_dim, *hidden, state_dim],
                              hidden_act="tanh", out_act="softmax", seed=seed)
        self.reward = MLP([state_dim + action_dim, *hidden, 1],
                          hidden_act="tanh", out_act=None, seed=seed + 1)
        self.trained = False

    def _feat(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(s), np.asarray(a)], axis=-1)

    def predict(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float]:
        x = self._feat(s, a).reshape(1, -1)
        sp = self.transition.predict(x).flatten()   # 下一步状态概率分布
        r = float(self.reward.predict(x)[0, 0])
        return sp, r

    def train(self, data: list, epochs: int = 100, lr: float = 0.05) -> dict:
        X = np.array([np.concatenate([s, a]) for s, a, _, _ in data])
        Ys = np.array([sp for _, _, sp, _ in data])          # one-hot 下一步
        Yr = np.array([[r] for _, _, _, r in data])
        loss_t = loss_r = float("nan")
        for ep in range(epochs):
            loss_t = self.transition.train_step(X, Ys, lr=lr, loss="ce")
            loss_r = self.reward.train_step(X, Yr, lr=lr, loss="mse")
        self.trained = True
        # 用交叉熵报告转移损失
        return {"epochs": epochs, "loss_transition": loss_t, "loss_reward": loss_r, "n": len(X)}

    def imagine(self, s0: np.ndarray, action_plan: list, horizon: int) -> Dict:
        """"想象"：不接触真实环境，用世界模型推演一条动作序列的结果。"""
        s = s0
        traj = []
        for t in range(horizon):
            a = action_plan[t % len(action_plan)] if action_plan else 0
            onehot = np.zeros(self.action_dim); onehot[a] = 1.0
            sp, r = self.predict(s, onehot)
            traj.append({"step": t, "action": a, "reward": r,
                         "next_argmax": int(np.argmax(sp))})
            s = np.clip(sp, 0.0, 1.0)
        return {"trajectory": traj, "total_reward": sum(x["reward"] for x in traj)}
