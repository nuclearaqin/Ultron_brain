"""导出 Ultron 神经组件权重为 GGUF 容器文件（"大脑权重包"）。

用法：
    python scripts/export_gguf.py [--out ultron_brain.gguf] [--train-link]
                                  [--force-train-link]

产物默认写到项目根目录 ultron_brain.gguf。

--train-link        链接预测器缺失，**或与当前嵌入维度不一致**时现场重训
--force-train-link  无论是否已有权重都重训（换嵌入模型后应显式跑一次）

注意（2026-10 修复）：旧版只在"权重文件不存在"时训练，于是换了 384 维语义
嵌入之后，仓库里那份**用 128 维哈希降级特征训练的**链接预测器会被一直沿用，
且导出后 npz 再被重训就会与 GGUF 不同步。现在维度不一致会直接重训并在导出后
做一致性自检，两个坑都有护栏。
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultron.controller_nn import BudgetController, ConvergenceDetector   # noqa: E402
from ultron.gguf_io import (check_file, export_brain, import_brain,       # noqa: E402
                            GGUFReader)
from ultron.graph import CausalGraph                                     # noqa: E402
from ultron.link_predictor import LinkPredictor, build_training_pairs    # noqa: E402

WEIGHTS_DIR = os.path.join(ROOT, "ultron", "weights")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out_path = os.path.join(ROOT, "ultron_brain.gguf")
    if "--out" in argv:
        out_path = argv[argv.index("--out") + 1]
    train_link = "--train-link" in argv or "--force-train-link" in argv
    force_train_link = "--force-train-link" in argv

    # ---- 1. 图统计 ----
    cg = CausalGraph(os.path.join(ROOT, "causal_graph.json"))
    cg.load()
    graph_stats = {"nodes": cg.node_count(), "edges": cg.edge_count(),
                   "source": "seed+conceptnet(zh)"}

    # ---- 2. 控制器 ----
    bc = BudgetController.load(os.path.join(WEIGHTS_DIR, "budget_controller.npz"))
    cd = ConvergenceDetector.load(os.path.join(WEIGHTS_DIR, "convergence_detector.npz"))
    if bc is None or cd is None:
        print("警告：控制器权重缺失，先运行 scripts/train_controllers.py")
        if bc is None and cd is None:
            return 1

    # ---- 3. 链接预测器 ----
    lp = LinkPredictor.load(os.path.join(WEIGHTS_DIR, "link_predictor.npz"))
    need_emb = lp is None or force_train_link
    emb = index = texts = embs = None
    if not need_emb:
        # 已有权重：仍要确认它与当前嵌入维度一致，否则训练/推理两侧会错配
        from ultron.embeddings import EmbeddingModel
        emb = EmbeddingModel()
        if lp.embed_dim != emb.dim:
            print(f"注意：链接预测器是 {lp.embed_dim} 维嵌入训练的，"
                  f"当前嵌入模型是 {emb.dim} 维。")
            if train_link:
                print("  -> 按 --train-link 重新训练。")
                need_emb = True
            else:
                print("  -> 仍按现有权重导出（加 --train-link 可重训）。")

    if need_emb:
        from ultron.embeddings import EmbeddingModel, NodeEmbeddingIndex
        if emb is None:
            emb = EmbeddingModel()
        index = NodeEmbeddingIndex(emb, os.path.join(ROOT, "causal_graph.json"))
        texts = cg.nodes()
        embs = index.encode_many(texts)
        print(f"训练链接预测器（嵌入 dim={emb.dim}，约 1~3 分钟）...")
        lp = LinkPredictor(emb.dim)
        pos, neg = build_training_pairs(texts, embs,
                                        [(u, v) for u, v in cg.graph.edges()],
                                        max_pos=20000, neg_ratio=2.0)
        info = lp.train(pos, neg, epochs=50)
        ev = lp.evaluate(pos, neg)
        lp.save(os.path.join(WEIGHTS_DIR, "link_predictor.npz"))
        print(f"  完成：loss={info['loss']:.4f} AUC={ev['auc']:.3f} "
              f"（正样本 {len(pos)} / 负样本 {len(neg)}）")
    if lp is None:
        print("警告：没有可用的链接预测器，将只导出控制器。")

    # ---- 4. 导出 ----
    summary = export_brain(
        out_path,
        budget_controller=bc,
        convergence_detector=cd,
        link_predictor=lp,
        graph_stats=graph_stats,
        embedding_info={"dim": getattr(lp, "embed_dim", None)},
    )
    print("=" * 60)
    print(f"已导出：{summary['path']}")
    print(f"  张量 {summary['tensors']} 个 / 元数据 {summary['metadata']} 项 / "
          f"对齐 {summary.get('alignment', 32)} 字节 / "
          f"{summary['bytes'] / 1024:.1f} KB")
    print(f"  组件：{', '.join(summary['components'])}")

    # ---- 5. 规范自检：对齐 / 越界 / 重叠 / 元数据维度 ----
    report = check_file(out_path)
    print(f"  规范自检（对齐/越界/重叠/元数据维度）："
          f"{'通过' if report['ok'] else '失败'}")
    for p in report["problems"]:
        print(f"    ! {p}")

    # ---- 6. 读回一致性 + 与磁盘权重同步性 ----
    # 同步性这一项是旧版缺失的护栏：GGUF 里的张量必须与 ultron/weights/*.npz
    # 当前内容一致，否则"大脑包"与配套权重是两代模型（曾实际发生过：
    # link_predictor.npz 在导出 5 分钟后被重训，GGUF 里留下上一代）。
    import numpy as np
    back = import_brain(out_path)
    sync_ok = True
    for tag, npz, prefix in (
            ("budget_controller", "budget_controller.npz", "budget_controller"),
            ("convergence_detector", "convergence_detector.npz", "convergence_detector"),
            ("link_predictor", "link_predictor.npz", "link_predictor")):
        path = os.path.join(WEIGHTS_DIR, npz)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        ref = back[tag]
        if ref is None:
            print(f"  {tag}：GGUF 内缺失（权重文件存在）-> 不同步")
            sync_ok = False
            continue
        worst = 0.0
        for i, (W, b) in enumerate(zip(ref.net.W, ref.net.b)):
            pW = np.asarray(data[f"W{i}"], dtype=float)
            pb = np.asarray(data[f"b{i}"], dtype=float)
            if pW.shape != W.shape or pb.shape != b.shape:
                worst = float("inf")
                break
            worst = max(worst, float(np.abs(W - pW).max()),
                        float(np.abs(b - pb).max()))
        good = worst < 1e-5
        sync_ok &= good
        print(f"  {tag}：与 {npz} 最大绝对误差 {worst:.2e} "
              f"{'一致' if good else '不一致（GGUF 已过期）'}")
    print(f"  读回一致性：{'通过' if sync_ok else '失败'}")
    if not (report["ok"] and sync_ok):
        return 1

    # ---- 7. 交叉验证：官方 gguf 包读取（如已安装） ----
    try:
        from gguf import GGUFReader as OfficialReader
        r = OfficialReader(out_path)
        names = [t.name for t in r.tensors]
        fields = {f.name: f for f in r.fields.values()}
        print(f"  官方 gguf 包读取：通过（识别张量 {len(names)} 个，"
              f"字段 {len(fields)} 项）")
        print(f"    样例张量：{names[:4]} ...")
    except ImportError:
        print("  官方 gguf 包未安装，跳过交叉验证（pip install gguf）")
    except Exception as e:
        print(f"  官方 gguf 包读取失败：{e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
