"""Ultron MCP Server：把推理协处理器暴露为 MCP 工具，供任意大模型调用。

协议：MCP（Model Context Protocol）JSON-RPC 2.0 over stdio，
兼容规范版本 2024-11-05。零第三方依赖实现（不需要官方 mcp SDK）。

工具清单（供大模型在推理流程中调用）：
  ultron_precheck  推理前：难度分级 / 思考预算 / 因果知识预取
  ultron_govern    思考中：收敛检测 / 重复识别 / 截停建议
  ultron_audit     答案前：因果主张一致性校验
  ultron_query     随时：因果知识查询（果/因/路径/干预）
  ultron_teach     随时：录入新因果边（需 --allow-write，默认只读禁用）

默认**只读**：协作方大模型不应改写本地知识库；仅当显式传入
`--allow-write` 时才开放 ultron_teach 的写入能力。

客户端配置示例（Claude Desktop / Cursor / WorkBuddy 等支持 MCP 的客户端）：

    {
      "mcpServers": {
        "ultron": {
          "command": "python",
          "args": ["-m", "ultron.mcp_server"],
          "cwd": "<项目根目录>"
        }
      }
    }

手动测试：
    echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python -m ultron.mcp_server
"""

import json
import sys
from typing import Any, Dict, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ultron"
SERVER_VERSION = "2.0.0"

TOOLS = [
    {
        "name": "ultron_precheck",
        "description": (
            "在开始思考/回答一个用户问题之前调用（推理预检）。"
            "返回：① 问题难度分级 difficulty（0=无需推理直接答，"
            "1=轻量推演，2=需要多步推理）；② 推荐思考预算 budget"
            "（max_steps 推理步数上限 / max_tokens 思考长度上限）；"
            "③ 本地因果知识库中与之相关的已知因果（knowledge）。"
            "请根据 budget 控制你的思考长度：difficulty=0 时不要展开"
            "长篇思考，difficulty=1 时简短推演，difficulty=2 时才深入推理。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "用户的原始问题"}
            },
            "required": ["question"],
        },
    },
    {
        "name": "ultron_govern",
        "description": (
            "在生成长思考（思维链）的过程中循环调用（思考治理）。"
            "把已生成的思考文本传入，检测是否已经收敛/陷入重复/超出预算。"
            "返回 stop=true 表示应立即停止思考并给出最终答案——"
            "继续思考只会浪费时间（overthinking）。"
            "建议每生成 2~4 段思考调用一次。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "原始问题"},
                "thinking": {"type": "string",
                             "description": "目前为止已生成的全部思考文本"},
                "steps_done": {"type": "integer",
                               "description": "已完成思考步数（可选）"},
                "budget": {"type": "object",
                           "description": "ultron_precheck 返回的预算（可选；"
                                          "不传则自动按深思考预算判断）"},
            },
            "required": ["thinking"],
        },
    },
    {
        "name": "ultron_audit",
        "description": (
            "在给出最终答案之前调用（因果审计）。把答案文本传入，"
            "校验其中每一条因果主张（A导致B）是否与本地因果知识库一致。"
            "返回每条主张的 verdict：supported（有直接因果）/ "
            "reachable（存在间接路径）/ unsupported（两概念均在库中但无路径）/ "
            "unknown（引用了库中没有的概念）。"
            "总体 score<0.5 时应检查答案是否有臆造的因果关系。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "原始问题（可选）"},
                "answer": {"type": "string", "description": "待审计的答案文本"},
            },
            "required": ["answer"],
        },
    },
    {
        "name": "ultron_query",
        "description": (
            "查询本地因果知识库（5.3 万节点中文因果图，毫秒级）。"
            "mode=effects 查「X 会导致什么」；mode=causes 查「Y 为什么发生」；"
            "mode=path 查两个概念的因果路径（需因果知识时优先用它，"
            "而不是用思维链去回忆）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string",
                         "enum": ["effects", "causes", "path"]},
                "node": {"type": "string", "description": "概念/现象（effects/causes 用）"},
                "cause": {"type": "string", "description": "起点（path 用）"},
                "effect": {"type": "string", "description": "终点（path 用）"},
            },
            "required": ["mode"],
        },
    },
    {
        "name": "ultron_teach",
        "description": (
            "向本地因果知识库录入一条新的因果边（当用户纠正或补充了"
            "因果关系时使用）。返回确认信息。注意：服务需以 --allow-write "
            "启动才可写入，默认只读模式下本工具返回禁用提示。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cause": {"type": "string", "description": "原因"},
                "effect": {"type": "string", "description": "结果"},
                "weight": {"type": "number",
                           "description": "强度 0~1（可选，默认 0.8）"},
            },
            "required": ["cause", "effect"],
        },
    },
]


class UltronMCPServer:
    """MCP stdio 服务：懒加载 bot，逐行处理 JSON-RPC 消息。

    write_enabled=False（默认）时服务只读：ultron_teach 会被拒绝。
    外部大模型默认不应改写本地知识库，写入能力须显式开启
    （启动参数 --allow-write），避免协作方模型误改/污染因果图。
    """

    def __init__(self, storage_path: str = "causal_graph.json",
                 write_enabled: bool = False):
        self.storage_path = storage_path
        self.write_enabled = bool(write_enabled)
        self.bot = None

    # ---------- 业务 ----------
    def _ensure_bot(self):
        if self.bot is None:
            from .bot import AdvancedCausalBot
            self.bot = AdvancedCausalBot(storage_path=self.storage_path,
                                         auto_think=False)
        return self.bot

    def _call_tool(self, name: str, args: Dict[str, Any]) -> str:
        bot = self._ensure_bot()
        cp = bot.coprocessor

        if name == "ultron_precheck":
            r = cp.precheck(str(args.get("question", "")))
            return json.dumps(r, ensure_ascii=False, indent=2)

        if name == "ultron_govern":
            q = str(args.get("question", ""))
            thinking = str(args.get("thinking", ""))
            steps = int(args.get("steps_done", 0) or 0)
            budget = args.get("budget")
            if not budget:
                budget = {"max_steps": 12, "max_tokens": 1024}
                if q:
                    try:
                        budget = cp.precheck(q)["budget"]
                    except Exception:
                        pass
            r = cp.govern(q, thinking, steps_done=steps,
                          budget=budget if isinstance(budget, dict) else None)
            return json.dumps(r, ensure_ascii=False, indent=2)

        if name == "ultron_audit":
            r = cp.audit(str(args.get("question", "")),
                         str(args.get("answer", "")))
            return json.dumps(r, ensure_ascii=False, indent=2)

        if name == "ultron_query":
            return self._query(args)

        if name == "ultron_teach":
            if not self.write_enabled:
                return ("写入已禁用：本服务以只读模式运行，外部模型不能改写"
                        "本地知识库。如需开启，请以 --allow-write 启动 "
                        "mcp_server。")
            cause = str(args.get("cause", "")).strip()
            effect = str(args.get("effect", "")).strip()
            weight = float(args.get("weight", 0.8) or 0.8)
            if not cause or not effect:
                return "失败：cause 与 effect 均不能为空。"
            existing = bot.cg.edge_info(cause, effect)
            if existing is not None and not args.get("overwrite"):
                return (f"已存在：{cause} -> {effect}"
                        f"（当前强度 {existing.get('weight', 1.0):.2f}）。"
                        f"如需覆盖请传 overwrite=true。")
            bot.add_causal_link(cause, effect, weight=weight,
                                overwrite=bool(args.get("overwrite")))
            bot._rebuild_node_index()
            return (f"已记录：{cause} -> {effect}（强度 {weight:.2f}）。"
                    f"当前图 {bot.cg.node_count()} 节点 / {bot.cg.edge_count()} 边。")

        raise ValueError(f"未知工具：{name}")

    def _query(self, args: Dict[str, Any]) -> str:
        bot = self._ensure_bot()
        mode = str(args.get("mode", "effects"))
        if mode == "effects":
            node = bot._match_arg(str(args.get("node", "")))
            if not node:
                return f"未在因果库中找到「{args.get('node', '')}」。"
            effs = bot.query_effects(node)
            return json.dumps({
                "node": node,
                "effects": [[v, round(w, 3)] for v, w, _ in effs[:12]],
                "total": len(effs),
            }, ensure_ascii=False, indent=2)
        if mode == "causes":
            node = bot._match_arg(str(args.get("node", "")))
            if not node:
                return f"未在因果库中找到「{args.get('node', '')}」。"
            causes = bot.query_causes(node)
            return json.dumps({
                "node": node,
                "causes": [[u, round(w, 3)] for u, w, _ in causes[:12]],
                "total": len(causes),
            }, ensure_ascii=False, indent=2)
        if mode == "path":
            a = bot._match_arg(str(args.get("cause", "")))
            b = bot._match_arg(str(args.get("effect", "")))
            if not a or not b:
                return f"节点匹配失败（cause={a}，effect={b}）。"
            path = bot.find_path(a, b)
            if not path:
                return json.dumps({"cause": a, "effect": b,
                                   "path": None,
                                   "note": "因果库中两节点之间无路径。"},
                                  ensure_ascii=False, indent=2)
            return json.dumps({
                "cause": a, "effect": b, "path": path,
                "strength": round(bot.path_strength(path), 3),
            }, ensure_ascii=False, indent=2)
        return f"未知 mode：{mode}（可选 effects/causes/path）"

    # ---------- 协议 ----------
    def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": (
                    "Ultron 是推理协处理器。推荐工作流：开始思考前调用 "
                    "ultron_precheck 获取思考预算；思考中每隔几步调用 "
                    "ultron_govern 检查是否可以停止；给出答案前调用 "
                    "ultron_audit 校验因果主张。需要因果事实时用 "
                    "ultron_query 直接查库，不要用思维链回忆。"
                ),
            }}

        if method in ("notifications/initialized", "notifications/cancelled",
                      "initialized"):
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": {"tools": TOOLS}}

        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                text = self._call_tool(name, args)
                return {"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": False}}
            except Exception as e:
                return {"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": f"工具执行失败：{e}"}],
                    "isError": True}}

        if msg_id is not None:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {
                "code": -32601, "message": f"Method not found: {method}"}}
        return None

    # ---------- 传输 ----------
    def run_stdio(self) -> int:
        """stdio 主循环。注意：业务日志一律走 stderr，stdout 只跑协议。"""
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

        real_stdout = sys.stdout
        sys.stdout = sys.stderr          # 屏蔽 bot 初始化期间的 print 污染

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle(msg)
            if resp is not None:
                real_stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                real_stdout.flush()
        return 0


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    storage = "causal_graph.json"
    if "--storage" in argv:
        try:
            storage = argv[argv.index("--storage") + 1]
        except Exception:
            pass
    return UltronMCPServer(storage_path=storage,
                           write_enabled=("--allow-write" in argv)).run_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
