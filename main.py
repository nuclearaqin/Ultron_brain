"""
奥创 Ultron —— 神经符号智能体（纯本地离线版）入口。

运行：
    python main.py              # 交互式对话
    python main.py --selftest   # 无交互自检：跑通 7 大能力（不加载语义模型也行）
    python main.py --demo       # 打印各能力的示范输出
    python main.py --serve      # 启动 HTTP REST API（默认 127.0.0.1:8600）
"""

import sys

# 修复：Windows 下当输出被重定向/管道捕获（如 IDE 运行、python main.py > log）时，
# Python 默认按 ANSI 代码页(GBK/cp936)编码 stdout，无法编码 ⏳✅🤖 等 Emoji 字符，
# 会在第一行 print 就抛 UnicodeEncodeError。这里强制切换为 UTF-8 并加 replace 兜底。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass

from ultron.bot import AdvancedCausalBot


def _coproc_selftest(bot) -> bool:
    """推理协处理器验收：PreCheck / Governor / Audit 三能力端到端。

    全部为确定性断言，不依赖外部大模型；任一断言失败则整体判定不通过。
    """
    cp = bot.coprocessor
    ok = True

    # ---- 1. PreCheck：难度分级应与任务的真实推理开销一致 ----
    cases = [("你好，请用一句话介绍你自己。", 0, "闲聊"),
             ("熬夜会怎样？", 1, "单跳因果"),
             ("如何一步步达成工作效率高？", 2, "多步规划"),
             # 回归（2026-10 复核）：_PLAN_KWS 曾漏"怎样/规划"，
             # 规划类问题被判 0 免思考；同时状态询问不能被误判为规划。
             ("怎样规划一周的饮食？", 2, "怎样式规划"),
             ("今天天气怎样？", 0, "状态询问非规划")]
    for q, expect, desc in cases:
        r = cp.precheck(q)
        good = r["difficulty"] == expect
        ok &= good
        print(f"      PreCheck「{q}」（{desc}）-> 难度 {r['difficulty']}"
              f" {r['difficulty_label']} / 预算 {r['budget']['max_tokens']} tok"
              f"  {'✓' if good else '✗ 期望 ' + str(expect)}")

    # ---- 2. Governor：冗余判停 / 未收敛不打断 / 超预算截停 ----
    roomy = {"max_steps": 12, "max_tokens": 1024}
    tight = {"max_steps": 3, "max_tokens": 64}
    rep_trace = "；".join(["熬夜会导致精神差"] * 5)
    checks = [
        ("冗余重复轨迹", cp.govern("熬夜会怎样", rep_trace, 0, roomy), True),
        ("初起轨迹（信息尚新）", cp.govern("熬夜会怎样", "先看看熬夜本身。", 0, roomy), False),
        ("超预算轨迹", cp.govern("熬夜会怎样", rep_trace, 0, tight), True),
    ]
    for label, g, expect_stop in checks:
        good = bool(g["stop"]) is expect_stop
        ok &= good
        print(f"      Governor {label}: stop={g['stop']} 把握={g['confidence']}"
              f"  {'✓' if good else '✗'}  {g['reason'][:30]}")

    # ---- 3. Audit：真主张 supported / 编造主张 unknown ----
    audit = cp.audit("", "熬夜会导致精神差；熬夜会导致黑眼圈；熬夜会导致地球爆炸")
    got = [c["verdict"] for c in audit["claims"]]
    expect = ["supported", "supported", "unknown"]
    good = got == expect
    ok &= good
    print(f"      Audit 主张校验：{[c['cause'] + '→' + c['effect'] + ':' + c['verdict'] for c in audit['claims']]}"
          f"  {'✓' if good else '✗ 期望 ' + str(expect)}")
    print(f"      审计结论：一致性={audit['consistent']} 得分={audit['score']}"
          f"（{audit['n_claims']} 条主张）")
    return ok


def selftest():
    """无交互自检：在不依赖真实语义模型的情况下，跑通 7 大能力。"""
    print("=" * 60)
    print("奥创 Ultron 自检 —— 验证 7 大升级目标")
    print("=" * 60)

    # 自检必须**完全不碰生产权重**：/learn 会把训练结果写进 weights_dir，
    # 2026-10 复核发现旧版自检在 6 节点示例图上训练后覆盖了生产
    # link_predictor.npz（8 条正样本盖掉 2 万条正样本的模型），也正是
    # "GGUF 与配套权重不同步"的真实成因。这里改用临时目录，退出即清理。
    import atexit
    import os as _os
    import shutil
    import tempfile
    _tmpdir = tempfile.mkdtemp(prefix="ultron_selftest_")
    atexit.register(shutil.rmtree, _tmpdir, ignore_errors=True)

    # 写入走临时目录（保护生产权重），读取仍指向包内生产目录——否则
    # 第 7 步测不到"规则+神经"融合路径，自检会静默降级成纯规则。
    _prod_weights = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "ultron", "weights")
    bot = AdvancedCausalBot(storage_path=_os.path.join(_tmpdir, "selftest.json"),
                            auto_think=False,
                            weights_dir=_os.path.join(_tmpdir, "weights"),
                            weights_read_dir=_prod_weights)

    # 1) 自监督学习 + 涌现泛化（链接预测）
    print("\n[1/7] 自监督学习 + 涌现泛化（链接预测）")
    bot.cg.add_causal_link("熬夜", "精神差", weight=0.8)
    bot.cg.add_causal_link("熬夜", "黑眼圈", weight=0.6)
    bot.cg.add_causal_link("精神差", "工作效率低", weight=0.7)
    bot._rebuild_node_index()
    print(bot._cmd_learn([]))
    print(bot._cmd_propose(["5"]))

    # 2) 推理：思维链 / 规划器 / test-time compute
    print("\n[2/7] 推理能力（CoT / 规划器 / test-time compute）")
    print(bot._cmd_reason(["", "下雨", "迟到"]))
    print(bot._cmd_plan(["", "迟到"]))
    print(bot._cmd_mc(["", "下雨", "迟到", "300"]))

    # 3) 世界模型 + 接地
    print("\n[3/7] 世界模型 + 接地（网格世界）")
    print(bot._cmd_world([]))

    # 4) 强化学习 + 自我对弈
    print("\n[4/7] 强化学习 + 自我对弈（井字棋）")
    print(bot._cmd_train(["", "150"]))

    # 5) 记忆 + RAG + 工具 + 反思回路
    print("\n[5/7] 智能体回路（检索 + 规划 + 工具 + 反思）")
    print(bot._cmd_agent(["", "下雨会导致什么后果"]))

    # 6) 神经-符号融合
    print("\n[6/7] 神经-符号融合现状")
    print(bot._cmd_fuse())

    # 7) 推理协处理器（对外协同的核心部件）
    print("\n[7/7] 推理协处理器（PreCheck / Governor / Audit）")
    coproc_ok = _coproc_selftest(bot)

    print("\n" + "=" * 60)
    print(f"自检完成。协处理器验收：{'通过' if coproc_ok else '未通过'}")
    print("=" * 60)
    # 临时图、嵌入缓存与权重都在 _tmpdir 内，atexit 已负责整体清理；
    # 同时顺手清掉旧版自检遗留在工作目录的文件（若有）。
    import os
    for tmp in ("ultron_selftest_tmp.json",):
        if os.path.exists(tmp):
            os.remove(tmp)
    for cache in ("ultron_selftest_tmp.json.emb_dim128.npz",
                  "ultron_selftest_tmp.json.emb_dim384.npz"):
        if os.path.exists(cache):
            os.remove(cache)


def demo():
    """打印各能力示范（不训练，快速展示）。"""
    bot = AdvancedCausalBot(storage_path="ultron_selftest_tmp.json",
                            auto_think=False)
    print("\n—— 演示：思维链 ——")
    print(bot._cmd_reason(["", "下雨", "迟到"]))
    print("\n—— 演示：规划器 ——")
    print(bot._cmd_plan(["", "迟到"]))
    print("\n—— 演示：智能体回路 ——")
    print(bot._cmd_agent(["", "为什么地面湿"]))
    import os
    for tmp in ("ultron_selftest_tmp.json",):
        if os.path.exists(tmp):
            os.remove(tmp)
    for cache in ("ultron_selftest_tmp.json.emb_dim128.npz",
                  "ultron_selftest_tmp.json.emb_dim384.npz"):
        if os.path.exists(cache):
            os.remove(cache)


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--demo" in sys.argv:
        demo()
        return
    if "--serve" in sys.argv or "--ui" in sys.argv:   # --ui 为旧名，保留兼容
        from ultron.server import serve
        port = 8600
        if "--port" in sys.argv:
            try:
                port = int(sys.argv[sys.argv.index("--port") + 1])
            except Exception:
                pass
        serve(storage_path="causal_graph.json", port=port)
        return
    if "--seed" in sys.argv:
        bot = AdvancedCausalBot(storage_path="causal_graph.json", auto_think=False)
        from ultron.seed_knowledge import install_seed
        n = install_seed(bot.cg, save=True)
        bot._rebuild_node_index()
        print(f"已安装内置中文常识因果种子库：新增 {n} 条。")
        print(f"当前图 {bot.cg.node_count()} 节点 / {bot.cg.edge_count()} 边。")
        return

    print(f"奥创 Ultron —— 神经符号因果智能体已就绪。")
    bot = AdvancedCausalBot(auto_think=True, think_interval=2.5)
    print("提示：直接问我「X会怎样 / 为什么Y」，或教我「X导致Y」。")
    print("输入 /help 查看全部能力（含 /learn /reason /plan /world /train /agent）。")
    print("我只会在此刻回答你；闲时我在独自推演。\n")
    try:
        while True:
            try:
                q = input("你 ▸ ").strip()
            except EOFError:
                break
            try:
                reply = bot.chat(q)
            except KeyboardInterrupt:
                break
            if reply is None:
                continue
            print(reply)
            if reply == "EXIT":
                break
    except KeyboardInterrupt:
        print("\n收到中断……")
    finally:
        print("\n先把这几轮的推演归档……")
        bot.stop_thinking()
        try:
            bot.cg.save()
            print("已保存全部因果。奥创，下线。")
        except Exception as e:
            print(f"警告：保存失败：{e}")


if __name__ == "__main__":
    main()
