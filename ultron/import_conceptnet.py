"""从 ConceptNet 导入现成的因果/常识知识（可选，需网络/数据文件）。

ConceptNet 5.7 是免费、多语言（含中文 zh）、CC-BY-SA 授权的常识语义网络，
其中 /r/Causes（导致）、/r/HasPrerequisite（先决条件）、/r/HasSubevent（子事件）
等关系可映射为奥创的"因→果"因果边。

用法：
    1) 下载官方 dump（约 2.5GB，含全部语言）：
       https://github.com/commonsense/conceptnet5/wiki/Downloads
       -> conceptnet-assertions-5.7.0.csv.gz
    2) 运行导入（流式过滤中文，只保留因果关系，限制条数控制规模）：
       python -m ultron.import_conceptnet path/to/conceptnet-assertions-5.7.0.csv.gz --max 3000

说明：官方 REST API（api.conceptnet.io）在国内常 502 不可达，故采用本地 CSV
导入；脚本支持 .csv 与 .csv.gz（自动解压）。
"""

import gzip
import json
import math
import re
import sys
import urllib.parse
from typing import Dict, List, Optional, Tuple

# 关系 -> (方向, 默认权重)：direction="start->end" 表示 start 是因
CAUSAL_RELATIONS = {
    "/r/Causes": ("start->end", 0.80),
    "/r/CausesDesire": ("start->end", 0.50),
    "/r/HasPrerequisite": ("end->start", 0.70),   # X 需要 Y 先发生 => Y 导致 X
    "/r/HasSubevent": ("start->end", 0.50),       # X 包含子事件 Y => X 弱导致 Y
    "/r/ObstructedBy": ("end->start", 0.50),      # X 被 Y 阻碍 => Y 反向影响 X
}

ZH_PATTERN = re.compile(r"/c/zh/")

# 噪声词表：疑问代词 / 因果逻辑词 / 泛化词 / 代词，作为节点会污染精确匹配
STOP_WORDS = {
    # 疑问代词
    "什么", "为什么", "怎么", "如何", "哪里", "哪个", "哪些", "谁", "啥", "咋",
    "怎么样", "怎样",
    # 因果/逻辑词
    "导致", "造成", "引起", "引发", "产生", "带来", "使得", "使", "让",
    "因为", "所以", "如果", "可能", "应该", "可以", "能够", "结果", "原因",
    "后果", "影响", "条件", "前提",
    # 泛化词/代词
    "东西", "问题", "事情", "情况", "状态", "程度", "方面", "方式", "方法",
    "时间", "地方", "时候", "人们", "大家", "我们", "他们", "你们", "自己",
    "别人", "有人", "一个人", "每个人", "很多", "一些", "所有", "一切",
    "比较", "非常", "各种", "其他", "另外", "同样", "不同", "一样", "这样",
    "那样", "这个", "那个", "一个", "一种",
}

# 繁体 -> 简体（ConceptNet 中文数据约 18% 为繁体，统一转简体避免节点分裂）
_CC = None
def _get_cc():
    global _CC
    if _CC is None:
        try:
            from opencc import OpenCC
            _CC = OpenCC("tw2sp")   # 台湾正体 -> 简体，能正确处理"著->着"等用字
        except Exception:
            _CC = False
    return _CC


def _parse_concept(uri: str) -> Optional[str]:
    """/c/zh/下雨 -> 下雨。繁转简 + 过滤噪声/单字节点。"""
    if not uri or ZH_PATTERN.search(uri) is None:
        return None
    term = uri.rsplit("/", 1)[-1]
    if "/" in term:            # 兼容 /c/zh/天气/n 这类带词性后缀
        term = term.split("/")[0]
    try:
        term = urllib.parse.unquote(term)
    except Exception:
        pass
    term = term.replace("_", " ").strip()
    if not term or len(term) > 24:
        return None
    cc = _get_cc()
    if cc:
        term = cc.convert(term)
    if len(term) < 2:          # 过滤单字节点（因果语义太弱且污染检索）
        return None
    if term in STOP_WORDS:     # 过滤疑问词/因果动词/泛化词
        return None
    return term


def _map_weight(raw: float) -> float:
    """ConceptNet 原始权重 -> 因果强度 [0,1]（对数映射）。

    ConceptNet 权重是"证据量"尺度（1.0=至少一个来源，2.0=两个来源或加权，
    更高=更多独立证据），近似 log2 增长。故用对数映射：

        raw=1.0 -> 0.45（单来源弱证据）
        raw=2.0 -> 0.61
        raw=4.0 -> 0.76
        raw=8.0 -> 0.92（封顶）

    相比旧线性映射（0.40 + 0.11*raw，cap 0.92）：旧版把 82% 的单来源边
    全挤在 0.51 一个格点上；对数映射让 [1,8] 区间单调铺开，并在弱证据段
    给出更诚实的低分（0.45 < 0.5，"有一定可能"而非"多半会"）。
    """
    raw = max(float(raw), 1.0)
    return min(0.92, 0.45 + 0.157 * math.log2(raw))


def parse_assertion_line(line: str) -> Optional[Tuple[str, str, str, float, float]]:
    """解析一行官方 CSV -> (因, 果, 关系, 映射权重, 原始权重) 或 None。"""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None
    uri, relation, start, end, meta = parts[0], parts[1], parts[2], parts[3], parts[4]
    if relation not in CAUSAL_RELATIONS:
        return None
    direction, default_w = CAUSAL_RELATIONS[relation]
    s = _parse_concept(start)
    e = _parse_concept(end)
    if not s or not e or s == e:
        return None
    raw = default_w
    try:
        raw = float(json.loads(meta).get("weight", default_w))
    except Exception:
        pass
    w = _map_weight(raw)
    if direction == "end->start":
        s, e = e, s
    return (s, e, relation, w, raw)


def _open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def import_csv(csv_path: str, graph, max_edges: Optional[int] = None,
               min_weight: float = 0.3) -> Dict:
    """流式导入 ConceptNet 中文因果边。

    max_edges=None 表示全量导入（不设上限）。返回统计。
    """
    stats = {"lines": 0, "zh_edges": 0, "imported": 0,
             "relations": {}, "duplicates": 0}
    batch: List[Tuple[str, str, float, dict]] = []
    with _open_maybe_gz(csv_path) as f:
        for line in f:
            stats["lines"] += 1
            if stats["lines"] % 5_000_000 == 0:
                print(f"    ... 已扫描 {stats['lines']:,} 行，"
                      f"命中中文因果边 {stats['zh_edges']:,}，"
                      f"已导入 {stats['imported']:,}", flush=True)
            if ZH_PATTERN.search(line) is None:   # 快速预过滤：跳过非中文行
                continue
            parsed = parse_assertion_line(line)
            if not parsed:
                continue
            cause, effect, relation, w, raw = parsed
            if w < min_weight:
                continue
            stats["zh_edges"] += 1
            stats["relations"][relation] = stats["relations"].get(relation, 0) + 1
            if graph.has_edge(cause, effect):
                stats["duplicates"] += 1
                continue
            batch.append((cause, effect, w, {},
                          {"source": "conceptnet", "relation": relation,
                           "raw_weight": raw}))
            if len(batch) >= 2000:
                stats["imported"] += graph.bulk_add_causal_links(batch)
                batch.clear()
            if max_edges is not None and stats["imported"] >= max_edges:
                break
    if batch:
        stats["imported"] += graph.bulk_add_causal_links(batch)
    return stats


def scan_only(csv_path: str, min_weight: float = 0.3) -> Dict:
    """只扫描统计，不写入图：统计中文因果边规模与关系分布。"""
    stats = {"lines": 0, "zh_edges": 0, "relations": {}}
    with _open_maybe_gz(csv_path) as f:
        for line in f:
            stats["lines"] += 1
            if ZH_PATTERN.search(line) is None:
                continue
            parsed = parse_assertion_line(line)
            if not parsed:
                continue
            _, _, relation, w, _ = parsed
            if w < min_weight:
                continue
            stats["zh_edges"] += 1
            stats["relations"][relation] = stats["relations"].get(relation, 0) + 1
    return stats


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 0
    csv_path = argv[0]
    max_edges = None           # 默认全量
    if "--max" in argv:
        try:
            max_edges = int(argv[argv.index("--max") + 1])
        except Exception:
            pass
    if "--scan" in argv:
        stats = scan_only(csv_path)
        print(f"扫描 {stats['lines']:,} 行，中文因果边 {stats['zh_edges']:,} 条。")
        for rel, n in sorted(stats["relations"].items(), key=lambda x: -x[1]):
            print(f"    {rel:<20} {n:,}")
        return 0

    from .graph import CausalGraph
    graph = CausalGraph("causal_graph.json")
    graph.load()
    print(f"开始导入（max={max_edges if max_edges is not None else '全量'}）...")
    stats = import_csv(csv_path, graph, max_edges=max_edges)
    graph.save()
    print(f"\n完成：扫描 {stats['lines']:,} 行，中文因果边 {stats['zh_edges']:,} 条，"
          f"去重 {stats['duplicates']:,}，导入 {stats['imported']:,} 条。")
    for rel, n in sorted(stats["relations"].items(), key=lambda x: -x[1]):
        print(f"    {rel:<20} {n:,}")
    print(f"当前图 {graph.node_count():,} 节点 / {graph.edge_count():,} 边。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
