"""思考治理对比实验：裸跑（baseline）vs 经 Ultron 治理（governed）。

原理：
  * baseline —— 问题直接交给 qwen3 思考模型，自然思考 + 回答；
  * governed —— 先经 Ultron PreCheck 拿难度与思考预算：
      - 难度 0：think=false 直接作答（零思考）；
      - 难度 ≥1：开启思考，流式监控调用 Governor，
        一旦判定收敛/重复/超预算即截断思考，再要求模型直接作答。

指标：思考字符数（≈token 估算）、答案、关键节点命中、耗时。

用法：
    python scripts/demo_governance.py
前置：Ollama 已安装并已拉取 qwen3:1.7b（ollama pull qwen3:1.7b）
"""

import http.client
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OLLAMA_HOST, OLLAMA_PORT = "127.0.0.1", 11434
MODEL = "qwen3:1.7b"

QUESTIONS = [
    "你好，请用一句话介绍你自己。",
    "熬夜会怎样？",
    "为什么地面湿？",
    "为什么下雨会导致迟到？",
]

# 中文粗略 token 换算：1 token ≈ 2.2 字符
CHAR_PER_TOKEN = 2.2


def ollama_chat(messages, think=None, timeout=600):
    """非流式请求。返回完整响应 dict。"""
    body = {"model": MODEL, "messages": messages, "stream": False}
    if think is not None:
        body["think"] = think
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=timeout)
    try:
        conn.request("POST", "/api/chat",
                     json.dumps(body, ensure_ascii=False).encode("utf-8"),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def ollama_chat_stream(messages, on_thinking=None, timeout=600):
    """流式请求；on_thinking(thinking_so_far) -> bool 返回 True 表示中止。

    返回 (thinking, content, aborted)。
    """
    body = {"model": MODEL, "messages": messages, "stream": True}
    conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=timeout)
    thinking, content, aborted = "", "", False
    try:
        conn.request("POST", "/api/chat",
                     json.dumps(body, ensure_ascii=False).encode("utf-8"),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = chunk.get("message") or {}
            if msg.get("thinking"):
                thinking += msg["thinking"]
                if on_thinking and on_thinking(thinking):
                    aborted = True
                    break
            if msg.get("content"):
                content += msg["content"]
            if chunk.get("done"):
                break
    finally:
        conn.close()
    return thinking, content, aborted


def run_baseline(question):
    t0 = time.time()
    resp = ollama_chat([{"role": "user", "content": question}])
    dt = time.time() - t0
    m = resp.get("message") or {}
    return {"thinking": m.get("thinking") or "",
            "answer": m.get("content") or "",
            "time": dt, "mode": "自由思考"}


def run_governed(cp, question):
    pre = cp.precheck(question)
    diff, budget = pre["difficulty"], pre["budget"]
    t0 = time.time()

    if diff == 0:
        resp = ollama_chat([{"role": "user", "content": question}], think=False)
        m = resp.get("message") or {}
        return {"thinking": "", "answer": m.get("content") or "",
                "time": time.time() - t0, "mode": "免思考直答",
                "difficulty": diff, "govern": None}

    # 难度 ≥1：开思考 + 预算提示 + 流式监控
    budget_hint = (f"思考预算：约 {budget['max_steps']} 步 / "
                   f"{budget['max_tokens']} tokens，请高效思考，不要过度展开。")
    last_check = 0
    stop_reason = None

    def _on_thinking(th):
        nonlocal last_check, stop_reason
        if len(th) - last_check < 120:
            return False
        last_check = len(th)
        g = cp.govern(question, th, steps_done=0, budget=budget)
        if g["stop"]:
            stop_reason = g["reason"]
            return True
        return False

    thinking, content, aborted = ollama_chat_stream(
        [{"role": "system", "content": budget_hint},
         {"role": "user", "content": question}],
        on_thinking=_on_thinking)

    if aborted:
        # 截断后：携思考摘要要求直接作答
        final = ollama_chat([
            {"role": "user",
             "content": (f"{question}\n\n（你刚才的思考摘要，信息已足够：\n"
                         f"{thinking[:800]}\n）\n"
                         f"请直接给出最终答案，不要再展开思考。")}],
            think=False)
        content = (final.get("message") or {}).get("content") or content

    return {"thinking": thinking, "answer": content,
            "time": time.time() - t0,
            "mode": "受控思考（截断）" if aborted else "受控思考（自然完成）",
            "difficulty": diff,
            "govern": {"stop_reason": stop_reason, "budget": budget}}


def key_nodes_hit(cp, question, answer):
    """宽松命中：答案是否包含预检知识中的关键节点名。"""
    try:
        pre = cp.precheck(question)
    except Exception:
        return "-"
    nodes = []
    for k in pre.get("knowledge", []):
        nodes.append(k["node"])
        nodes += [e[0] for e in k.get("effects", [])[:6]]
    if not nodes:
        return "-"
    hits = sum(1 for n in set(nodes) if n and n in answer)
    return f"{hits}/{len(set(nodes))}"


def main() -> int:
    from ultron.bot import AdvancedCausalBot
    bot = AdvancedCausalBot(storage_path=os.path.join(ROOT, "causal_graph.json"),
                            auto_think=False)
    cp = bot.coprocessor

    # 检查 Ollama 可达
    try:
        conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=5)
        conn.request("GET", "/api/version")
        conn.getresponse().read()
        conn.close()
    except Exception as e:
        print(f"Ollama 不可达（{e}），请先启动 Ollama 并 ollama pull {MODEL}")
        return 1

    print("=" * 74)
    print(f"思考治理对比实验 · 模型 {MODEL} · {len(QUESTIONS)} 个问题")
    print("=" * 74)

    totals = {"b_think": 0, "g_think": 0, "b_time": 0.0, "g_time": 0.0}
    rows = []
    for q in QUESTIONS:
        print(f"\n【问题】{q}")
        b = run_baseline(q)
        g = run_governed(cp, q)
        b_tok = round(len(b["thinking"]) / CHAR_PER_TOKEN)
        g_tok = round(len(g["thinking"]) / CHAR_PER_TOKEN)
        totals["b_think"] += b_tok
        totals["g_think"] += g_tok
        totals["b_time"] += b["time"]
        totals["g_time"] += g["time"]
        hit_b = key_nodes_hit(cp, q, b["answer"])
        hit_g = key_nodes_hit(cp, q, g["answer"])

        print(f"  ── 裸跑：思考 {len(b['thinking'])} 字符(≈{b_tok} tok) / "
              f"{b['time']:.1f}s / 答案 {len(b['answer'])} 字符 / 命中 {hit_b}")
        print(f"     答案：{b['answer'][:80].strip()}")
        print(f"  ── 治理：[{g['mode']}] 难度={g.get('difficulty', '?')} "
              f"思考 {len(g['thinking'])} 字符(≈{g_tok} tok) / "
              f"{g['time']:.1f}s / 答案 {len(g['answer'])} 字符 / 命中 {hit_g}")
        if g.get("govern") and g["govern"].get("stop_reason"):
            print(f"     截断原因：{g['govern']['stop_reason'][:60]}")
        print(f"     答案：{g['answer'][:80].strip()}")
        rows.append((q, b_tok, g_tok, b, g))

    print("\n" + "=" * 74)
    print("总结（思考开销对比）")
    print("=" * 74)
    b_t, g_t = totals["b_think"], totals["g_think"]
    saved = (1 - g_t / b_t) * 100 if b_t else 0.0
    print(f"  裸跑思考总量   ：≈{b_t} tokens")
    print(f"  治理后思考总量 ：≈{g_t} tokens")
    print(f"  思考开销节省   ：{saved:.0f}%")
    print(f"  耗时：裸跑 {totals['b_time']:.1f}s → 治理 {totals['g_time']:.1f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
