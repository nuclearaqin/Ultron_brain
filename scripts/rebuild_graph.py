"""重建因果图：种子库 + ConceptNet 全量重导入（带来源元数据与对数权重映射）。

背景：旧图（v1）的边权重来自线性映射 0.40+0.11*raw，82% 的单来源边
全挤在 0.51 一个格点上，且无法区分边的来源（种子/ConceptNet/手工）。
本脚本从原始数据重建：

  1. 备份现有 causal_graph.json 到 backups/
  2. 从空图开始：安装内置种子库（source=seed）
  3. 全量导入 ConceptNet 中文因果边（source=conceptnet，保存 raw_weight）
  4. 保存并打印统计（节点/边/权重分布/来源分布）

用法：
    python scripts/rebuild_graph.py [--csv path/to/conceptnet-assertions-5.7.0.csv.gz]
"""

import os
import shutil
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultron.graph import CausalGraph              # noqa: E402
from ultron.import_conceptnet import import_csv   # noqa: E402
from ultron.seed_knowledge import install_seed    # noqa: E402


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    csv_path = os.path.join(ROOT, "data", "conceptnet-assertions-5.7.0.csv.gz")
    if "--csv" in argv:
        csv_path = argv[argv.index("--csv") + 1]
    graph_path = os.path.join(ROOT, "causal_graph.json")

    if not os.path.exists(csv_path):
        print(f"找不到 ConceptNet 数据文件：{csv_path}")
        return 1

    # ---- 1. 备份 ----
    if os.path.exists(graph_path):
        backup_dir = os.path.join(ROOT, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"causal_graph.v1.{stamp}.json")
        shutil.copy2(graph_path, backup_path)
        print(f"[备份] {graph_path} -> {backup_path}")

    # ---- 2. 重建 ----
    cg = CausalGraph(graph_path)
    t0 = time.time()
    n_seed = install_seed(cg, save=False)
    print(f"[种子] 安装内置种子库：{n_seed} 条（source=seed）")

    print(f"[导入] 从 {os.path.basename(csv_path)} 全量导入（约需数分钟）...")
    stats = import_csv(csv_path, cg, max_edges=None, min_weight=0.3)
    cg.save()
    t1 = time.time()

    # ---- 3. 统计 ----
    weights = [d.get("weight", 1.0) for _, _, d in cg.graph.edges(data=True)]
    src = Counter(d.get("source", "manual") for _, _, d in cg.graph.edges(data=True))
    print(f"\n[完成] 耗时 {t1 - t0:.1f}s")
    print(f"  图规模：{cg.node_count()} 节点 / {cg.edge_count()} 边")
    print(f"  扫描 {stats['lines']:,} 行，命中中文因果边 {stats['zh_edges']:,}，"
          f"去重 {stats['duplicates']:,}，导入 {stats['imported']:,}")

    import numpy as np
    ws = np.array(weights)
    print(f"  权重：min={ws.min():.3f} mean={ws.mean():.3f} max={ws.max():.3f} "
          f"std={ws.std():.3f}（v1 的 std=0.084，越大约有区分度）")
    print(f"  来源分布：{dict(src)}")
    for rel, n in sorted(stats["relations"].items(), key=lambda x: -x[1]):
        print(f"    {rel:<22} {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
