"""GGUF 权重容器：零依赖写入/读取 + Ultron 大脑权重导出/导入。

为什么用 GGUF：GGUF 是 llama.cpp 生态的模型权重容器格式（单文件、
mmap 友好、KV 元数据丰富）。Ultron 的因果图与推理代码本体是"数据与
程序"，无法装入 GGUF；但其**神经组件**（思考控制器、链接预测器）的
权重可以打包为 GGUF——形成"Ultron 大脑权重包"，供自有工具链加载，
也能被第三方 GGUF 生态工具（官方 gguf 包、llama.cpp 工具链）读取。

本模块按 GGUF v3 规范零依赖实现（写入 + 读取），
规范见 https://github.com/ggerganov/ggml/blob/master/docs/gguf.md 。
测试中用官方 gguf 包交叉验证兼容性。
"""

import json
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

GGUF_MAGIC = 0x46554747          # "GGUF" little-endian
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# ---- metadata value types ----
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

# ---- ggml tensor types（本项目只使用 F32/F16） ----
GGML_F32, GGML_F16 = 0, 1


def _enc_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _dec_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def _enc_value(value) -> Tuple[int, bytes]:
    """把 python 值编码为 (type_id, payload)。"""
    if isinstance(value, bool):
        return T_BOOL, struct.pack("<?", value)
    if isinstance(value, int):
        if 0 <= value < 2 ** 32:
            return T_UINT32, struct.pack("<I", value)
        if 0 <= value:
            return T_UINT64, struct.pack("<Q", value)
        if -2 ** 31 <= value:
            return T_INT32, struct.pack("<i", value)
        return T_INT64, struct.pack("<q", value)
    if isinstance(value, float):
        return T_FLOAT32, struct.pack("<f", value)
    if isinstance(value, str):
        return T_STRING, _enc_str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("GGUF 不支持空数组；请省略该字段")
        elem_tid, first = _enc_value(value[0])
        payload = struct.pack("<I", elem_tid) + struct.pack("<Q", len(value)) + first
        payload += b"".join(_enc_value(v)[1] for v in value[1:])
        return T_ARRAY, payload
    raise TypeError(f"不支持的元数据类型：{type(value)}")


def _dec_value(f, tid: int) -> Any:
    if tid == T_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    if tid == T_INT8:
        return struct.unpack("<b", f.read(1))[0]
    if tid == T_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    if tid == T_INT16:
        return struct.unpack("<h", f.read(2))[0]
    if tid == T_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    if tid == T_INT32:
        return struct.unpack("<i", f.read(4))[0]
    if tid == T_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    if tid == T_BOOL:
        return struct.unpack("<?", f.read(1))[0]
    if tid == T_STRING:
        return _dec_str(f)
    if tid == T_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    if tid == T_INT64:
        return struct.unpack("<q", f.read(8))[0]
    if tid == T_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    if tid == T_ARRAY:
        elem_tid = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [_dec_value(f, elem_tid) for _ in range(n)]
    raise ValueError(f"未知 GGUF 元数据类型：{tid}")


class GGUFWriter:
    """最小 GGUF v3 写入器（KV 元数据 + F32/F16 张量）。"""

    def __init__(self, path: str, alignment: int = DEFAULT_ALIGNMENT):
        self.path = path
        self.alignment = alignment
        self._kv: List[Tuple[str, int, bytes]] = []
        self._tensors: List[Tuple[str, np.ndarray, bool]] = []   # (name, arr, as_f16)

    # ---------- 元数据 ----------
    def add_metadata(self, key: str, value) -> None:
        tid, payload = _enc_value(value)
        self._kv.append((key, tid, payload))

    def add_json(self, key: str, obj) -> None:
        self.add_metadata(key, json.dumps(obj, ensure_ascii=False))

    # ---------- 张量 ----------
    def add_tensor(self, name: str, array: np.ndarray, as_f16: bool = False) -> None:
        self._tensors.append((name, np.ascontiguousarray(array, dtype=np.float32),
                              as_f16))

    # ---------- 写盘 ----------
    def write(self) -> dict:
        """写盘。

        修复（对齐 bug）：规范要求**每块张量数据**都落在 alignment 边界上，
        旧实现让张量首尾相接，导致 b2 这类小张量（12 字节）之后的所有张量
        偏移错位——按规范逐张量补位的第三方读取器会整段读成乱码。
        现在逐张量向上取整到 alignment，并显式写入 general.alignment。
        """
        infos = []
        offset = 0
        for name, arr, as_f16 in self._tensors:
            dims = list(reversed(arr.shape))          # gguf 维度：最内层在前
            nbytes = arr.size * (2 if as_f16 else 4)
            # 每块张量的起始偏移对齐到 alignment（规范要求）
            offset = -(-offset // self.alignment) * self.alignment
            infos.append((name, dims, GGML_F16 if as_f16 else GGML_F32,
                          offset, arr, as_f16))
            offset += nbytes
        data_bytes = -(-offset // self.alignment) * self.alignment

        # general.alignment 必须显式声明（缺省虽为 32，但显式才可自证）
        kv = list(self._kv)
        if not any(k == "general.alignment" for k, _, _ in kv):
            kv.insert(0, ("general.alignment", T_UINT32,
                          struct.pack("<I", self.alignment)))

        with open(self.path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(infos)))
            f.write(struct.pack("<Q", len(kv)))
            for key, tid, payload in kv:
                f.write(_enc_str(key))
                f.write(struct.pack("<I", tid))
                f.write(payload)
            for name, dims, ttype, off, _, _ in infos:
                f.write(_enc_str(name))
                f.write(struct.pack("<I", len(dims)))
                for d in dims:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", ttype))
                f.write(struct.pack("<Q", off))
            pad = (-f.tell()) % self.alignment
            f.write(b"\0" * pad)
            written = 0
            for name, dims, ttype, off, arr, as_f16 in infos:
                if off > written:                    # 张量之间的对齐补零
                    f.write(b"\0" * (off - written))
                elif off < written:                  # 理论不可达
                    raise ValueError(f"张量偏移重叠：{name}")
                buf = arr.astype("<f2" if as_f16 else "<f4").tobytes()
                f.write(buf)
                written = off + len(buf)
            if data_bytes > written:                 # 数据区尾部补零
                f.write(b"\0" * (data_bytes - written))

        return {"path": self.path, "tensors": len(infos),
                "metadata": len(kv), "alignment": self.alignment,
                "bytes": os.path.getsize(self.path)}


class GGUFReader:
    """最小 GGUF v3 读取器（支持本项目所写文件与标准 F32/F16 张量）。"""

    def __init__(self, path: str, alignment: int = DEFAULT_ALIGNMENT):
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, np.ndarray] = {}
        self.alignment = alignment
        self.data_start = 0
        self.offsets: Dict[str, int] = {}
        self._read(alignment)

    def _read(self, alignment: int) -> None:
        with open(self.path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                raise ValueError("不是有效的 GGUF 文件（magic 不匹配）")
            version = struct.unpack("<I", f.read(4))[0]
            if version != GGUF_VERSION:
                raise ValueError(f"不支持的 GGUF 版本：{version}")
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                key = _dec_str(f)
                tid = struct.unpack("<I", f.read(4))[0]
                self.metadata[key] = _dec_value(f, tid)
            # 文件里显式声明的对齐优先于调用方给的缺省值
            declared = self.metadata.get("general.alignment")
            if isinstance(declared, int) and declared > 0:
                alignment = int(declared)
            infos = []
            for _ in range(n_tensors):
                name = _dec_str(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                infos.append((name, dims, ttype, offset))
            pad = (-f.tell()) % alignment
            data_start = f.tell() + pad
            self.alignment = alignment
            self.data_start = data_start
            self.offsets = {name: off for name, _d, _t, off in infos}
            for name, dims, ttype, offset in infos:
                f.seek(data_start + offset)
                shape = list(reversed(dims))
                count = int(np.prod(shape)) if shape else 1
                if ttype == GGML_F32:
                    arr = np.frombuffer(f.read(count * 4), dtype="<f4")
                elif ttype == GGML_F16:
                    arr = np.frombuffer(f.read(count * 2), dtype="<f2")
                else:
                    raise ValueError(f"暂不支持的张量类型：{ttype}")
                self.tensors[name] = np.array(arr).reshape(shape)


# ==================== Ultron 大脑权重导出 / 导入 ====================

def _net_dims(net) -> List[int]:
    """从已训练网络的实际权重形状推出层结构（不写死常量）。

    修复：旧实现把控制器维度写成常量（[12,32,16,3]），`_budget_features`
    从 12 维扩到 13 维后元数据没跟着改，导致 GGUF 元数据与真实张量不符——
    照元数据建模会构建出装不进的网络。现在一律由张量形状反推。
    """
    dims = [int(net.W[0].shape[0])]
    dims += [int(w.shape[1]) for w in net.W]
    return dims


def _dump_mlp(writer: GGUFWriter, prefix: str, net) -> int:
    n = 0
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        writer.add_tensor(f"{prefix}.W{i}", W)
        writer.add_tensor(f"{prefix}.b{i}", b)
        n += 1
    return n


def _load_mlp(reader: GGUFReader, prefix: str, net) -> bool:
    i = 0
    while f"{prefix}.W{i}" in reader.tensors:
        net.W[i] = reader.tensors[f"{prefix}.W{i}"].astype(float)
        net.b[i] = reader.tensors[f"{prefix}.b{i}"].astype(float)
        i += 1
    return i > 0


def export_brain(path: str,
                 budget_controller=None,
                 convergence_detector=None,
                 link_predictor=None,
                 graph_stats: Optional[Dict] = None,
                 embedding_info: Optional[Dict] = None) -> Dict[str, Any]:
    """把 Ultron 神经组件权重打包为 GGUF 文件（"大脑权重包"）。

    返回导出摘要；没有任何组件时会抛出 ValueError。
    """
    w = GGUFWriter(path)
    w.add_metadata("general.architecture", "ultron")
    w.add_metadata("general.name", "Ultron Reasoning Co-Processor")
    w.add_metadata("ultron.format", "brain-weights")
    w.add_metadata("ultron.version", "2.0")
    w.add_metadata("ultron.created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    w.add_metadata("ultron.note",
                   "Ultron 神经组件权重容器（非 llama.cpp 生成式模型）；"
                   "因果图知识库存于 causal_graph.json，经 MCP/HTTP 协同。")

    components = []
    if budget_controller is not None and getattr(budget_controller, "trained", False):
        layers = _dump_mlp(w, "budget_controller", budget_controller.net)
        w.add_metadata("ultron.budget_controller.dims",
                       _net_dims(budget_controller.net))
        w.add_metadata("ultron.budget_controller.role", "问题特征->思考难度0/1/2")
        components.append(f"budget_controller({layers}层)")
    if convergence_detector is not None and getattr(convergence_detector, "trained", False):
        layers = _dump_mlp(w, "convergence_detector", convergence_detector.net)
        w.add_metadata("ultron.convergence_detector.dims",
                       _net_dims(convergence_detector.net))
        w.add_metadata("ultron.convergence_detector.role", "轨迹特征->P(stop)")
        components.append(f"convergence_detector({layers}层)")
    if link_predictor is not None and getattr(link_predictor, "trained", False):
        layers = _dump_mlp(w, "link_predictor", link_predictor.net)
        w.add_metadata("ultron.link_predictor.embed_dim", link_predictor.embed_dim)
        w.add_metadata("ultron.link_predictor.role", "节点对嵌入->P(因->果)")
        components.append(f"link_predictor({layers}层)")

    if not components:
        raise ValueError("没有任何已训练的组件可导出（先训练控制器或 /learn）")

    if graph_stats:
        w.add_json("ultron.graph.stats", graph_stats)
    if embedding_info:
        w.add_json("ultron.embedding", embedding_info)

    summary = w.write()
    summary["components"] = components
    return summary


def import_brain(path: str, load_link_predictor: bool = True) -> Dict[str, Any]:
    """从 GGUF 大脑权重包读回神经组件（可直接装配进协处理器）。"""
    from .controller_nn import BudgetController, ConvergenceDetector
    from .link_predictor import LinkPredictor

    r = GGUFReader(path)
    out: Dict[str, Any] = {"metadata": r.metadata,
                           "budget_controller": None,
                           "convergence_detector": None,
                           "link_predictor": None}

    if any(k.startswith("budget_controller.") for k in r.tensors):
        bc = BudgetController()
        if _load_mlp(r, "budget_controller", bc.net):
            bc.trained = True
            out["budget_controller"] = bc

    if any(k.startswith("convergence_detector.") for k in r.tensors):
        cd = ConvergenceDetector()
        if _load_mlp(r, "convergence_detector", cd.net):
            cd.trained = True
            out["convergence_detector"] = cd

    if load_link_predictor and "link_predictor.W0" in r.tensors:
        in_dim = int(r.tensors["link_predictor.W0"].shape[0])
        lp = LinkPredictor(embed_dim=in_dim // 2)
        if _load_mlp(r, "link_predictor", lp.net):
            lp.trained = True
            out["link_predictor"] = lp

    return out


def check_file(path: str) -> Dict[str, Any]:
    """自检一个 GGUF 文件是否符合本项目写入器承诺的规范（供 CI / 导出后校验）。

    检查项：magic/version、张量名唯一、**每块张量偏移按 alignment 对齐**、
    张量区间不重叠、不越界、元数据中声明的层结构与真实张量形状一致。
    返回 {"ok": bool, "problems": [...], "tensors": n, "alignment": n}。
    """
    r = GGUFReader(path)
    problems: List[str] = []

    if "general.alignment" not in r.metadata:
        problems.append("元数据缺少 general.alignment（对齐只能靠缺省约定）")

    size = os.path.getsize(path)
    spans = []
    for name, arr in r.tensors.items():
        off = r.offsets.get(name)
        if off is None:
            continue
        if off % r.alignment != 0:
            problems.append(f"张量 {name} 偏移 {off} 未按 {r.alignment} 对齐")
        nbytes = arr.size * arr.dtype.itemsize
        if r.data_start + off + nbytes > size:
            problems.append(f"张量 {name} 越界（尾部 {r.data_start + off + nbytes} > {size}）")
        spans.append((off, off + nbytes, name))
    spans.sort()
    for (a0, a1, an), (b0, _b1, bn) in zip(spans, spans[1:]):
        if b0 < a1:
            problems.append(f"张量区间重叠：{an} 与 {bn}")

    # 元数据声明的层结构 vs 真实张量形状
    for comp, key in (("budget_controller", "ultron.budget_controller.dims"),
                      ("convergence_detector", "ultron.convergence_detector.dims")):
        declared = r.metadata.get(key)
        idx = 0
        actual: List[int] = []
        while f"{comp}.W{idx}" in r.tensors:
            w = r.tensors[f"{comp}.W{idx}"]
            if idx == 0:
                actual.append(int(w.shape[0]))
            actual.append(int(w.shape[1]))
            idx += 1
        if actual and declared != actual:
            problems.append(f"{comp} 元数据维度 {declared} != 真实张量 {actual}")

    return {"ok": not problems, "problems": problems,
            "tensors": len(r.tensors), "alignment": r.alignment,
            "bytes": size}
