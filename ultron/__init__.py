"""
奥创 Ultron —— 神经符号智能体（纯本地离线版）

一个以"因果图(符号) + 语义嵌入(神经)"为核心、并具备以下能力的自研智能体：
  1. 规模化自监督学习 + 涌现式泛化（链接预测 / 概念抽象）
  2. 推理能力：test-time compute、思维链(CoT)、规划器
  3. 世界模型 + 接地（可学习的环境转移模型 T(s,a)->s'）
  4. 强化学习与自我对弈（策略梯度，涌现策略）
  5. 记忆 + 检索增强(RAG) + 工具使用 + 反思回路（真正的智能体）
  6. 可组合、可泛化的神经-符号表示（neuro-symbolic）

设计原则：神经组件全部用纯 numpy 实现，SentenceTransformer 语义模型可选（懒加载 + 哈希降级），
无 GPU / 无网络也能完整运行。
"""

import sys

# 包级编码修复：无论从哪个入口（python main.py / 直接 import ultron / 测试）
# 使用本包，都强制 stdout/stderr 为 UTF-8，避免中文 Windows(GBK) 下
# print('⏳') 抛 UnicodeEncodeError。errors='replace' 保证任何环境都不会崩溃。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass

__version__ = "2.0.0"
__all__ = [
    "nn", "config", "graph", "embeddings", "link_predictor",
    "reasoner", "world_model", "rl", "memory", "tools", "agent", "bot",
]
