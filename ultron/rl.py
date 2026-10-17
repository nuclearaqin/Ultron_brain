"""强化学习 + 自我对弈（目标 4）：由奖励/目标驱动，涌现策略。

用纯 numpy 的策略网络（含价值头），在井字棋里自我对弈：
同一条网络同时执黑/执白，胜利 +1、失败 -1、平局 0，REINFORCE + 价值基线
更新策略。训练后网络学会"合理落子"，即从奖励信号中涌现的对弈策略。
"""

from typing import Dict, List, Tuple

import numpy as np

from .nn import MLP


class TicTacToe:
    """井字棋环境。state: 9 维（1=我方, -1=对方, 0=空）。"""

    LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros(9, dtype=float)
        self.done = False
        self.winner = 0
        return self.board.copy()

    def legal_actions(self) -> List[int]:
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action: int, player: int) -> Tuple[np.ndarray, float, bool]:
        assert self.board[action] == 0
        self.board[action] = player
        self._check(player)
        if self.winner != 0:
            self.done = True
            return self.board.copy(), 1.0, True
        if len(self.legal_actions()) == 0:
            self.done = True
            return self.board.copy(), 0.0, True
        return self.board.copy(), 0.0, False

    def _check(self, player: int) -> None:
        for a, b, c in self.LINES:
            if self.board[a] == self.board[b] == self.board[c] == player:
                self.winner = player
                return


class PolicyNet:
    """策略网络：输入 9 维棋盘（相对当前玩家视角），输出 9 维动作 logits + 价值。"""

    def __init__(self, seed: int = 0):
        self.policy = MLP([9, 64, 64, 9], hidden_act="tanh", out_act="softmax", seed=seed)
        self.value = MLP([9, 64, 64, 1], hidden_act="tanh", out_act=None, seed=seed + 100)

    def act(self, board: np.ndarray, legal: List[int], temperature: float = 1.0) -> Tuple[int, float]:
        x = board.reshape(1, -1)
        logits = self.policy.predict(x).flatten()
        # 屏蔽非法动作
        mask = np.full(9, -1e9)
        for a in legal:
            mask[a] = logits[a]
        if temperature <= 0:
            a = int(np.argmax(mask))
        else:
            p = np.exp((mask - np.max(mask)) / temperature)
            p = p / p.sum()
            a = int(np.random.choice(9, p=p))
        v = float(self.value.predict(x)[0, 0])
        return a, v

    def value_of(self, board: np.ndarray) -> float:
        return float(self.value.predict(board.reshape(1, -1))[0, 0])

    def train_batch(self, states: List[np.ndarray], actions: List[int],
                    returns: List[float], lr: float = 0.01) -> Dict:
        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(returns).reshape(-1, 1)
        # 价值基线
        baseline = self.value.predict(states)
        advantage = returns - baseline
        # 策略梯度：dL/dlogit = advantage * (pi - onehot)
        pi, caches = self.policy.forward(states)
        y = np.zeros_like(pi)
        y[np.arange(len(actions)), actions] = 1.0
        d_policy = (pi - y) * advantage
        self.policy.zero_grad()
        self.policy.backward(d_policy, caches, pi)
        self.policy.step(lr)
        # 价值损失
        self.value.train_step(states, returns, lr=lr, loss="mse")
        return {"mean_return": float(np.mean(returns)),
                "mean_advantage": float(np.mean(np.abs(advantage)))}


class SelfPlay:
    """自我对弈训练器：两方共用同一 PolicyNet。"""

    def __init__(self, env: TicTacToe, net: PolicyNet, seed: int = 0):
        self.env = env
        self.net = net
        self.rng = np.random.default_rng(seed)

    def play_game(self, temperature: float = 1.0) -> List[dict]:
        """一盘自我对弈，返回 (state(相对视角), action, reward) 轨迹。"""
        self.env.reset()
        states, actions, players = [], [], []
        player = 1
        while not self.env.done:
            board = self.env.board
            legal = self.env.legal_actions()
            # 相对当前玩家视角（己方恒为 +1）
            view = board * player
            a, _ = self.net.act(view, legal, temperature=temperature)
            states.append(view)
            actions.append(a)
            players.append(player)
            _, _, _ = self.env.step(a, player)
            player = -player
        winner = self.env.winner
        out = []
        for s, a, p in zip(states, actions, players):
            r = 1.0 if winner == p else (-1.0 if winner == -p else 0.0)
            out.append({"state": s, "action": a, "reward": r})
        return out

    def train(self, games: int = 300, lr: float = 0.02, temperature: float = 1.0) -> Dict:
        history = []
        for g in range(games):
            traj = self.play_game(temperature)
            states = [t["state"] for t in traj]
            actions = [t["action"] for t in traj]
            returns = [t["reward"] for t in traj]
            if states:
                self.net.train_batch(states, actions, returns, lr=lr)
            history.append(traj[0]["reward"] if traj else 0.0)
        return {
            "games": games,
            "first_player_avg": float(np.mean(history)),
            "win_rate_vs_random": self.eval_vs_random(50),
        }

    def eval_vs_random(self, games: int = 50) -> float:
        wins = 0
        for _ in range(games):
            self.env.reset()
            player = 1
            while not self.env.done:
                if player == 1:
                    view = self.env.board
                    a, _ = self.net.act(view, self.env.legal_actions(), temperature=0.1)
                else:
                    a = int(self.rng.choice(self.env.legal_actions()))
                _, _, _ = self.env.step(a, player)
                player = -player
            if self.env.winner == 1:
                wins += 1
        return wins / games
