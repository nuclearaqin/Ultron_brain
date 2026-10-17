"""奥创 HTTP 后端：纯标准库 REST 服务（零新增依赖，纯本地离线）。

只提供 API（不含任何网页/前端资源），供编排框架、脚本或其他大模型调用：
  GET  /           -> 端点索引（纯 JSON）
  GET  /api/state  -> 图统计 / 内推轮次 / 思考流 / 知识缺口 / 疑点
  POST /api/chat   -> {"text": "..."} => {"reply": "...", ...}
  POST /api/precheck | /api/govern | /api/audit | /api/query
                   -> 推理协处理器（见 coprocessor.py）

启动：
  python main.py --serve         # 推荐
  python main.py --ui            # 旧名，等价
  python -m ultron.server --port 8600
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .bot import AdvancedCausalBot
from .config import BOT_NAME

# API 端点索引（GET / 返回；纯 JSON，不涉任何网页）
API_INDEX = {
    "service": "ultron",
    "description": "推理协处理器：预检 / 思考治理 / 因果审计 / 因果查询",
    "endpoints": {
        "GET  /api/state": "运行状态与图统计",
        "POST /api/chat": "对话式因果问答",
        "POST /api/precheck": "推理前：难度分级 + 思考预算 + 因果知识预取",
        "POST /api/govern": "思考中：收敛/重复检测与截停建议",
        "POST /api/audit": "答案前：因果主张一致性校验",
        "POST /api/query": "因果知识查询（effects / causes / path）",
    },
}


class _State:
    """持有 bot 单例，供 handler 线程访问（bot 内部有 RLock，线程安全）。"""

    def __init__(self, storage_path: str, think_interval: float = 2.5):
        self.bot = AdvancedCausalBot(storage_path=storage_path,
                                     auto_think=True,
                                     think_interval=think_interval)
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        bot = self.bot
        with bot._lock:
            nodes = bot.cg.node_count()
            edges = bot.cg.edge_count()
        return {
            "name": BOT_NAME,
            "nodes": nodes,
            "edges": edges,
            "think_count": bot._think_count,
            "embedding": ("sentence-transformers" if bot.emb.available
                          else "hash-fallback"),
            "embedding_dim": bot.emb.dim,
            "thoughts": list(bot.thoughts)[-30:],
            "gaps": list(bot._gaps)[-8:],
            "suspicions": list(bot._suspicions)[-8:],
            "dialog": {
                "turn": bot.dialog_state.get("turn", 0),
                "topic": bot.dialog_state.get("topic"),
                "last_intent": bot.dialog_state.get("last_intent"),
            },
        }


def make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        server_version = "UltronAPI/1.0"

        # ---- 工具 ----
        def _send_json(self, obj: dict, code: int = 200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):   # 静音访问日志
            pass

        # ---- 路由 ----
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                self._send_json(state.snapshot())
            elif path in ("/", "/index.html"):
                self._send_json(API_INDEX)     # 纯 JSON 端点索引，无网页
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/chat":
                self._api_chat()
            elif path in ("/api/precheck", "/api/govern", "/api/audit",
                          "/api/query"):
                self._api_coproc(path)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        # ---- 请求体 ----
        def _read_payload(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8", errors="replace")
                return json.loads(raw) if raw.strip() else {}
            except Exception:
                return None

        # ---- 旧接口：/api/chat ----
        def _api_chat(self):
            payload = self._read_payload()
            if payload is None:
                self._send_json({"error": "bad request"}, 400)
                return
            text = str(payload.get("text", "")).strip()
            if not text:
                self._send_json({"error": "empty text"}, 400)
                return

            with state.lock:   # chat 会改对话状态/可能教学写图，串行化
                try:
                    reply = state.bot.chat(text)
                except Exception as e:
                    self._send_json({"error": f"chat failed: {e}"}, 500)
                    return

            self._send_json({
                "reply": reply if reply is not None else "",
                "nodes": state.bot.cg.node_count(),
                "edges": state.bot.cg.edge_count(),
                "think_count": state.bot._think_count,
            })

        # ---- 推理协处理器接口（供外部大模型/编排框架调用） ----
        def _api_coproc(self, path: str):
            payload = self._read_payload()
            if payload is None:
                self._send_json({"error": "bad request"}, 400)
                return
            bot = state.bot
            cp = bot.coprocessor
            try:
                with state.lock:
                    if path == "/api/precheck":
                        q = str(payload.get("question", "")).strip()
                        if not q:
                            self._send_json({"error": "empty question"}, 400)
                            return
                        r = cp.precheck(q)
                    elif path == "/api/govern":
                        thinking = str(payload.get("thinking", ""))
                        if not thinking.strip():
                            self._send_json({"error": "empty thinking"}, 400)
                            return
                        q = str(payload.get("question", ""))
                        steps = int(payload.get("steps_done", 0) or 0)
                        budget = payload.get("budget")
                        if not isinstance(budget, dict):
                            budget = (cp.precheck(q)["budget"] if q.strip()
                                      else {"max_steps": 12, "max_tokens": 1024})
                        r = cp.govern(q, thinking, steps_done=steps,
                                      budget=budget)
                    elif path == "/api/audit":
                        answer = str(payload.get("answer", ""))
                        if not answer.strip():
                            self._send_json({"error": "empty answer"}, 400)
                            return
                        r = cp.audit(str(payload.get("question", "")), answer)
                    else:  # /api/query
                        r = self._query(payload)
                self._send_json(r)
            except Exception as e:
                self._send_json({"error": f"coprocessor failed: {e}"}, 500)

        def _query(self, payload: dict) -> dict:
            bot = state.bot
            mode = str(payload.get("mode", "effects"))
            if mode == "effects":
                node = bot._match_arg(str(payload.get("node", "")))
                if not node:
                    return {"error": "node not found",
                            "hint": "试试更常见的说法，或先 /seed 安装种子知识"}
                effs = bot.query_effects(node)
                return {"node": node,
                        "effects": [[v, round(w, 3)] for v, w, _ in effs[:12]],
                        "total": len(effs)}
            if mode == "causes":
                node = bot._match_arg(str(payload.get("node", "")))
                if not node:
                    return {"error": "node not found"}
                causes = bot.query_causes(node)
                return {"node": node,
                        "causes": [[u, round(w, 3)] for u, w, _ in causes[:12]],
                        "total": len(causes)}
            if mode == "path":
                a = bot._match_arg(str(payload.get("cause", "")))
                b = bot._match_arg(str(payload.get("effect", "")))
                if not a or not b:
                    return {"error": "node not found", "cause": a, "effect": b}
                path = bot.find_path(a, b)
                if not path:
                    return {"cause": a, "effect": b, "path": None,
                            "note": "因果库中两节点之间无路径。"}
                return {"cause": a, "effect": b, "path": path,
                        "strength": round(bot.path_strength(path), 3)}
            return {"error": f"unknown mode: {mode}",
                    "hint": "可选 effects / causes / path"}

    return Handler


def serve(storage_path: str = "causal_graph.json", host: str = "127.0.0.1",
          port: int = 8600, think_interval: float = 2.5) -> None:
    state = _State(storage_path, think_interval=think_interval)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    base = f"http://{host}:{port}"
    print("🌐 HTTP API 已启动（纯 REST，无网页）：")
    print(f"   · 端点索引：{base}/")
    print(f"   · 运行状态：{base}/api/state")
    print("   Ctrl+C 停止服务（退出时会保存因果图）。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n正在关闭：停止内推线程并保存因果图……")
        state.bot.stop_thinking()
        try:
            state.bot.cg.save()
        except Exception as e:
            print(f"警告：保存失败：{e}")
        httpd.server_close()
        print("已保存。奥创，下线。")


def main(argv: Optional[list] = None) -> int:
    argv = argv or []
    import sys
    args = argv if argv else sys.argv[1:]
    port = 8600
    storage = "causal_graph.json"
    if "--port" in args:
        try:
            port = int(args[args.index("--port") + 1])
        except Exception:
            pass
    if "--storage" in args:
        try:
            storage = args[args.index("--storage") + 1]
        except Exception:
            pass
    serve(storage_path=storage, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
