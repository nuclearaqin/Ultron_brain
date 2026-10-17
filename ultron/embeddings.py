"""语义嵌入层（神经半边的基础）：SentenceTransformer 懒加载 + 确定性哈希降级。

国内网络环境直连 huggingface 常失败，故：
  1. 默认走 hf-mirror.com 镜像；
  2. 加载失败时自动降级到基于字符 n-gram 的确定性哈希特征，保证离线也能跑。
"""

import os
import threading
import zlib
from typing import List, Optional, Tuple

import numpy as np

from .config import MODEL_LOAD_TIMEOUT, MODEL_NAME

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _hash_embed(text: str, dim: int = 128) -> np.ndarray:
    """字符级 n-gram 特征哈希（无网络、跨进程稳定），作为语义模型的兜底。"""
    t = (text or "").lower()
    v = np.zeros(dim, dtype=float)
    grams = list(t)
    grams += [t[i:i + 2] for i in range(len(t) - 1)]
    grams += [t[i:i + 3] for i in range(len(t) - 2)]
    for g in grams:
        v[zlib.crc32(g.encode('utf-8')) % dim] += 1.0
    norm = np.linalg.norm(v)
    if norm > 1e-9:
        v /= norm
    return v


class EmbeddingModel:
    """统一嵌入接口。encode() 返回 [n, dim] 矩阵。"""

    def __init__(self, model_name: str = MODEL_NAME, dim: int = 128):
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self.available = False
        self._load()

    def _load(self) -> None:
        """带超时保护地加载语义模型：有缓存时秒载；无网无缓存时 25s 内降级哈希，
        不会让整个程序无限期挂在网络重试上。本会话内不切换（保证索引一致）。"""
        result = {"model": None, "error": None}
        done = threading.Event()

        def worker():
            try:
                from sentence_transformers import SentenceTransformer
                result["model"] = SentenceTransformer(self.model_name)
            except Exception as e:
                result["error"] = e
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        ok = done.wait(timeout=MODEL_LOAD_TIMEOUT)

        if ok and result["model"] is not None:
            self._model = result["model"]
            self.available = True
            try:
                self.dim = int(np.asarray(self._model.encode(["测试"])).shape[1])
            except Exception:
                pass
            print(f"[嵌入] 语义模型已就绪（维度 {self.dim}）。")
        elif ok and result["error"] is not None:
            print(f"[嵌入] 语义模型不可用，降级为哈希特征（维度 {self.dim}）：{result['error']}")
        else:
            print(f"[嵌入] 语义模型加载超时（>{MODEL_LOAD_TIMEOUT:.0f}s），"
                  f"本会话降级为哈希特征（维度 {self.dim}）。"
                  f"模型一旦下载完成，下次启动将立即加载。")

    def encode(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.available and self._model is not None:
            try:
                return np.asarray(self._model.encode(list(texts)), dtype=float)
            except Exception:
                pass
        return np.array([_hash_embed(t, self.dim) for t in texts], dtype=float)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """余弦相似度。a: [m,d], b: [n,d] -> [m,n]。"""
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
        return a @ b.T

    # ==================== 节点索引 ====================
    def index_nodes(self, node_texts: List[str]) -> np.ndarray:
        if not node_texts:
            return np.array([])
        return self.encode(node_texts)

    def best_matches(self, phrase: str, node_texts: List[str],
                     node_embeddings: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """返回与 phrase 最相似的 k 个节点（完全包含匹配优先）。"""
        if not node_texts or node_embeddings.size == 0:
            return []
        phrase = phrase.strip()
        if not phrase:
            return []
        exact = [n for n in node_texts if n in phrase]
        emb = self.encode([phrase])
        sims = self.similarity(emb, node_embeddings).flatten()
        ranked = sorted(zip(node_texts, sims.tolist()), key=lambda x: -x[1])
        out: List[Tuple[str, float]] = []
        seen = set()
        for n in exact + [r[0] for r in ranked]:
            if n not in seen and len(out) < k:
                seen.add(n)
                sim = 1.0 if n in exact else dict(ranked).get(n, 0.0)
                out.append((n, sim))
        return out


class NodeEmbeddingIndex:
    """持久化 + 增量的节点嵌入索引。

    大图（数万节点）每次启动全量编码会耗时 1 分钟以上；此索引把嵌入落盘
    （{stem}.emb_dim{n}.npz），启动只加载缓存；新增节点只增量编码，
    内存仅保留 float32 矩阵，供 bot / memory / agent 共享复用。
    """

    def __init__(self, embeddings: EmbeddingModel, cache_stem: str):
        self.emb = embeddings
        # 缓存键含维度：真实模型(384)与哈希降级(128)互不混淆
        self.path = f"{cache_stem}.emb_dim{embeddings.dim}.npz"
        self._keys: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        """读回缓存。

        修复（可移植性）：旧实现用 `dtype=object` 存 keys，numpy 会把它写成
        pickle——文件名里没有 .pkl，但 `np.load(..., allow_pickle=False)`
        （安全扫描器与不少加载器的默认值）会直接拒绝。现在优先按无 pickle
        读取，读到旧格式则读后自动重写成定长字符串数组，一次自愈。
        """
        data = None
        legacy = False
        for allow_pickle in (False, True):
            try:
                data = np.load(self.path, allow_pickle=allow_pickle)
                keys = [str(k) for k in np.asarray(data["keys"]).tolist()]
                mat = np.asarray(data["emb"], dtype=np.float32)
                legacy = np.asarray(data["keys"]).dtype == object
                break
            except Exception:
                data = None
        if data is None:
            self._keys = []
            self._matrix = None
            return
        if len(keys) == mat.shape[0] and mat.shape[1] == self.emb.dim:
            self._keys = keys
            self._matrix = mat
            if legacy:
                print("[嵌入缓存] 检测到旧版 pickle 格式，重写为无 pickle 格式。")
                self._save()
            return
        self._keys = []
        self._matrix = None

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            # keys 用定长 Unicode 数组：不触发 pickle，allow_pickle=False 可读
            np.savez(self.path, keys=np.array(self._keys, dtype=str),
                     emb=self._matrix)
        except Exception as e:
            print(f"[嵌入缓存] 写入失败（不影响运行）：{e}")

    def encode_many(self, texts: List[str]) -> np.ndarray:
        """返回与 texts 同序的嵌入矩阵；缺失节点增量编码并落盘。"""
        if not texts:
            return np.array([])
        keymap = {t: i for i, t in enumerate(self._keys)}
        missing = [t for t in texts if t not in keymap]
        if missing:
            new_emb = np.asarray(self.emb.encode(missing), dtype=np.float32)
            if self._matrix is None:
                self._matrix = new_emb
            else:
                self._matrix = np.vstack([self._matrix, new_emb])
            self._keys.extend(missing)
            self._save()
        if self._matrix is None:            # 理论不可达：texts 非空至少有一次写入
            return np.zeros((len(texts), self.emb.dim), dtype=np.float32)
        keymap = {t: i for i, t in enumerate(self._keys)}
        idxs = [keymap[t] for t in texts]
        return self._matrix[idxs]
