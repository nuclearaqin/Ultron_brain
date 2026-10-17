"""接口回归：HTTP REST API + MCP Server 端到端可用性检查。

大模型协同的两条通道都必须可被"外部"调用，本脚本以真实子进程 + 真实
网络/管道请求的形态验证，而非进程内函数调用。

用法：
    python scripts/check_interfaces.py [--port 8611] [--skip-chat]

检查项：
    HTTP  GET  /api/state        服务与图规模快照
    HTTP  POST /api/precheck     难度分级 + 预算 + 知识预取
    HTTP  POST /api/govern       思考轨迹判停
    HTTP  POST /api/audit        因果主张校验
    HTTP  POST /api/query        因果库直查
    HTTP  OPTIONS *              CORS 预检
    HTTP  POST /api/chat         对话入口（可用 --skip-chat 跳过）
    MCP   initialize / tools/list / 5 个 tools/call（默认只读模式）
    MCP   写路径隔离验证（临时图 + --allow-write，测完即删）

退出码：全部通过 0，任一失败 1。
"""

import http.client
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PY = sys.executable
RESULTS = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'通过' if ok else '失败'}] {name}" + (f" —— {detail}" if detail else ""))
    return ok


# ==================== HTTP ====================

def http_json(port: int, method: str, path: str, body=None, timeout: float = 90.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        # 注意：http.client 对 str 类型 body 会按 latin-1 编码，中文必先转
        # UTF-8 bytes，否则抛 UnicodeEncodeError。
        payload = (None if body is None
                   else json.dumps(body, ensure_ascii=False).encode("utf-8"))
        headers = {"Content-Type": "application/json"} if payload else {}
        conn.request(method, path, payload, headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        try:
            return resp.status, json.loads(raw)
        except json.JSONDecodeError:
            return resp.status, raw
    finally:
        conn.close()


def wait_http(port: int, timeout: float = 120.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            status, data = http_json(port, "GET", "/api/state", timeout=5)
            if status == 200 and isinstance(data, dict) and "nodes" in data:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def check_http(port: int, skip_chat: bool) -> None:
    print("\n—— HTTP REST API ——")
    status, data = http_json(port, "GET", "/api/state")
    record("GET /api/state", status == 200 and isinstance(data, dict),
           f"图 {data.get('nodes')} 节点 / {data.get('edges')} 边"
           if isinstance(data, dict) else str(data)[:60])

    status, data = http_json(port, "POST", "/api/precheck",
                             {"question": "你好，请用一句话介绍你自己。"})
    ok = status == 200 and data.get("difficulty") == 0
    record("POST /api/precheck（闲聊→免思考）", ok,
           f"difficulty={data.get('difficulty')} budget={data.get('budget')}"
           if isinstance(data, dict) else str(data)[:60])

    status, data = http_json(port, "POST", "/api/precheck",
                             {"question": "为什么下雨会导致迟到？"})
    ok = status == 200 and data.get("difficulty", 0) >= 1
    record("POST /api/precheck（因果→给预算）", ok,
           f"difficulty={data.get('difficulty')} label={data.get('difficulty_label')}"
           if isinstance(data, dict) else str(data)[:60])

    status, data = http_json(port, "POST", "/api/govern", {
        "question": "熬夜会怎样",
        "thinking": "；".join(["熬夜会导致精神差"] * 5),
        "budget": {"max_steps": 12, "max_tokens": 1024}})
    ok = status == 200 and data.get("stop") is True
    record("POST /api/govern（冗余→判停）", ok,
           f"stop={data.get('stop')} 把握={data.get('confidence')}"
           if isinstance(data, dict) else str(data)[:60])

    # 审计：真主张 supported / 图中不存在的概念 unknown
    # 注："地球爆炸" 在 ConceptNet 中真实存在（概念已知但无因果路径 -> unsupported），
    # 故用确定不存在的名词验证 unknown 分支。
    status, data = http_json(port, "POST", "/api/audit",
                             {"answer": "下雨会导致土石流；熬夜会导致紫电幻兽"})
    got = [c["verdict"] for c in data.get("claims", [])] if isinstance(data, dict) else []
    ok = status == 200 and got == ["supported", "unknown"]
    record("POST /api/audit（真主张 supported / 编造概念 unknown）", ok,
           f"verdicts={got}")

    status, data = http_json(port, "POST", "/api/query",
                             {"mode": "effects", "node": "下雨"})
    ok = status == 200 and data.get("node") == "下雨" and data.get("total", 0) > 0
    record("POST /api/query（查后果）", ok,
           f"{data.get('total')} 条结果，样例 "
           f"{[e[0] for e in data.get('effects', [])[:3]]}"
           if isinstance(data, dict) else str(data)[:60])

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("OPTIONS", "/api/precheck")
        resp = conn.getresponse()
        cors = resp.getheader("Access-Control-Allow-Origin")
        resp.read()
        conn.close()
        record("OPTIONS（CORS 预检）", resp.status == 204 and cors == "*",
               f"status={resp.status} allow-origin={cors}")
    except Exception as e:
        record("OPTIONS（CORS 预检）", False, str(e)[:60])

    if skip_chat:
        print("  [跳过] POST /api/chat（--skip-chat）")
    else:
        status, data = http_json(port, "POST", "/api/chat", {"text": "你好"},
                                 timeout=180)
        ok = status == 200 and isinstance(data, dict) and "reply" in data
        record("POST /api/chat（对话入口）", ok,
               f"reply={str(data.get('reply', ''))[:40]!r}"
               if isinstance(data, dict) else str(data)[:60])


# ==================== MCP ====================

def mcp_session(requests, timeout: float = 180.0, extra_args=None):
    """启动 MCP stdio 服务，逐条发送请求，返回应答列表。"""
    cmd = [PY, "-u", "-m", "ultron.mcp_server"] + list(extra_args or [])
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    out = []
    try:
        for req in requests:
            proc.stdin.write((json.dumps(req, ensure_ascii=False) + "\n")
                             .encode("utf-8"))
            proc.stdin.flush()
        t0 = time.time()
        while len(out) < len(requests) and time.time() - t0 < timeout:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line.decode("utf-8", errors="replace")))
            except json.JSONDecodeError:
                continue
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return out


def check_mcp() -> None:
    print("\n—— MCP Server（stdio / JSON-RPC 2.0）——")
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "regression", "version": "1.0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "ultron_precheck",
                    "arguments": {"question": "为什么下雨会导致迟到？"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "ultron_govern",
                    "arguments": {"question": "熬夜会怎样",
                                  "thinking": "；".join(["熬夜会导致精神差"] * 5)}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "ultron_audit",
                    "arguments": {"answer": "下雨会导致土石流；熬夜会导致紫电幻兽"}}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "ultron_query",
                    "arguments": {"mode": "effects", "node": "下雨"}}},
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
         "params": {"name": "ultron_teach",
                    "arguments": {"cause": "接口回归测试因", "effect": "接口回归测试果"}}},
    ]
    resps = {r.get("id"): r for r in mcp_session(reqs)}
    if not record("MCP 握手（7 条请求均有应答）", len(resps) == len(reqs),
                  f"收到 {len(resps)}/{len(reqs)} 条"):
        return

    init = resps[1].get("result", {})
    record("initialize（协议版本与服务器信息）",
           init.get("protocolVersion") == "2024-11-05"
           and "serverInfo" in init,
           f"{init.get('serverInfo', {}).get('name')} "
           f"v{init.get('serverInfo', {}).get('version')}")

    tools = resps[2].get("result", {}).get("tools", [])
    names = [t["name"] for t in tools]
    record("tools/list（5 个工具）", len(tools) == 5, "、".join(names))

    def tool_text(idx):
        content = resps[idx].get("result", {}).get("content", [])
        return content[0]["text"] if content else ""

    def tool_json(idx):
        try:
            return json.loads(tool_text(idx))
        except Exception:
            return {}

    pre = tool_json(3)
    record("tools/call ultron_precheck", pre.get("difficulty", -1) >= 1,
           f"difficulty={pre.get('difficulty')} "
           f"budget={pre.get('budget')}")

    gov = tool_json(4)
    record("tools/call ultron_govern", gov.get("stop") is True,
           f"stop={gov.get('stop')} 把握={gov.get('confidence')}")

    aud = tool_json(5)
    record("tools/call ultron_audit",
           [c["verdict"] for c in aud.get("claims", [])] == ["supported", "unknown"],
           f"verdicts={[c['verdict'] for c in aud.get('claims', [])]}")

    q = tool_json(6)
    record("tools/call ultron_query",
           q.get("node") == "下雨" and q.get("total", 0) > 0,
           f"{q.get('total')} 条结果，样例 "
           f"{[e[0] for e in q.get('effects', [])[:3]]}")

    # 默认只读：外部模型的写入请求必须被拒绝（保护本地知识库）
    teach_text = tool_text(7)
    record("tools/call ultron_teach（只读模式下拒绝写入）",
           "写入已禁用" in teach_text, teach_text[:60])


def check_mcp_write(tmp_dir: str) -> None:
    """写路径隔离验证：临时小图上开启 --allow-write，确认 teach 真的写入。"""
    print("\n—— MCP 写路径（隔离临时图 + --allow-write）——")
    storage = os.path.join(tmp_dir, "mcp_write_test.json")
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "ultron_teach",
                    "arguments": {"cause": "回归测试因", "effect": "回归测试果",
                                  "weight": 0.7}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "ultron_query",
                    "arguments": {"mode": "effects", "node": "回归测试因"}}},
    ]
    resps = {r.get("id"): r for r in mcp_session(
        reqs, extra_args=["--storage", storage, "--allow-write"])}

    def text(idx):
        content = resps.get(idx, {}).get("result", {}).get("content", [])
        return content[0]["text"] if content else ""

    ok_write = len(resps) == 2 and "已记录" in text(1)
    record("ultron_teach 写入成功（--allow-write）", ok_write,
           text(1).strip().splitlines()[0][:60] if text(1) else "无应答")

    try:
        q = json.loads(text(2))
    except Exception:
        q = {}
    record("写入后可查到该因果（临时图）",
           q.get("node") == "回归测试因" and q.get("total", 0) >= 1,
           f"{q.get('total')} 条结果：{[e[0] for e in q.get('effects', [])[:3]]}")

    # 清理临时文件（含嵌入缓存）
    for f in os.listdir(tmp_dir):
        if f.startswith("mcp_write_test"):
            try:
                os.remove(os.path.join(tmp_dir, f))
            except OSError:
                pass


def main() -> int:
    argv = sys.argv[1:]
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8611
    skip_chat = "--skip-chat" in argv

    print("=" * 70)
    print("Ultron 接口回归：HTTP REST API + MCP Server")
    print("=" * 70)

    server = subprocess.Popen(
        [PY, "-u", "-m", "ultron.server", "--port", str(port), "--no-browser"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if wait_http(port):
            check_http(port, skip_chat)
        else:
            record("HTTP 服务启动", False, f"{port} 端口在超时内未就绪")
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()

    check_mcp()

    import tempfile
    with tempfile.TemporaryDirectory(prefix="ultron_regression_") as tmp:
        check_mcp_write(tmp)

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"结果：{passed}/{total} 项通过")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  ✗ {name} —— {detail}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
