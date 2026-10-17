# Ultron · 推理协处理器（Reasoning Co-Processor）

> 给大模型配一个**本地推理副驾**：思考前先分级、思考中判停、出答案前核验。
> 实测思考开销 **↓75%**（1049 → 267 tokens），关键因果节点命中率不变。

大模型负责语言与生成，Ultron 负责"便宜的推理"。

## 它解决什么问题

推理型大模型（DeepSeek-R1 / QwQ / qwen3-thinking 等）普遍存在**思考过剩
（overthinking）**：简单问候也可能思考数百字，单跳常识反复验证——这是当前
LLM 推理最大的算力浪费源之一。Ultron 以**本地毫秒级**的因果推理引擎 +
微型神经控制器，为大模型提供三件它做不到的事：

| 能力 | 调用时机 | 作用 |
|------|---------|------|
| **PreCheck 预检** | 思考开始前 | 问题难度分级（0 免思考 / 1 轻推演 / 2 深推演）、思考预算建议、因果知识预取 |
| **Governor 治理** | 思考过程中 | 滚动检测收敛度/重复度/预算消耗 → 给出停止或继续的建议（截停 overthinking） |
| **Audit 审计** | 给出答案前 | 抽取因果主张并对照因果知识库校验，标注每条主张的可信等级 |

内核是一个 **5.3 万节点 / 11.5 万条边**的中文因果知识图（内置常识种子库 +
ConceptNet 中文因果导入），配合两个轻量神经控制器（预算 / 收敛检测，
纯 numpy 训练，可导出 GGUF 权重容器）。

* **零重依赖**：核心仅需 numpy + networkx；无 `sentence-transformers` 时
  自动降级为哈希嵌入，功能完整、精度下降。
* **三处接入**：MCP Server（stdio）/ HTTP REST API / CLI——纯服务端接口，
  不捆绑任何前端页面，也不绑定某一家模型。

## 快速开始

```bash
# 1. 安装核心依赖（零重依赖，可选装语义模型）
pip install -r requirements.txt
# 推荐（可选）：真实语义嵌入，未装时自动降级为哈希特征
pip install sentence-transformers

# 2. 训练思考控制器（约 10 秒；权重写入 ultron/weights/）
python scripts/train_controllers.py

# 3. 启动（三选一）
python main.py --serve             # HTTP REST API (127.0.0.1:8600，纯接口无网页)
python -m ultron.mcp_server        # MCP Server（stdio，供 Claude/Cursor 等）
python main.py                     # 交互式对话 CLI
```

> 已有 `causal_graph.json`（53,204 节点 / 114,951 边）开箱即用；
> 如需从 ConceptNet 原始数据重建（带来源标注与对数权重映射）：
> `python scripts/rebuild_graph.py`（约 70 秒）。

## 接入方式

### 1. MCP Server（推荐，任何支持 MCP 的客户端可直接挂载）

```json
{
  "mcpServers": {
    "ultron": {
      "command": "python",
      "args": ["-m", "ultron.mcp_server"],
      "cwd": "<本项目根目录>"
    }
  }
}
```

暴露 5 个工具：`ultron_precheck` / `ultron_govern` / `ultron_audit` /
`ultron_query` / `ultron_teach`。推荐工作流（已写入 initialize 的 instructions）：

```
开始思考前 → ultron_precheck（拿到预算与知识，按难度决定想多少）
思考中     → ultron_govern（每 2~4 段调用，stop=true 立即收束作答）
给出答案前 → ultron_audit（校验因果主张）
需要事实时 → ultron_query（直接查库，别用思维链回忆）
```

> **默认只读**：`ultron_teach` 会改写本地知识库，协作方模型不应默认具备该
> 能力，故默认拒绝写入；确需开启时以 `python -m ultron.mcp_server
> --allow-write` 启动。

### 2. HTTP REST API（供任意编排框架/脚本调用）

| 端点 | 方法 | 请求体 | 说明 |
|------|------|--------|------|
| `/api/precheck` | POST | `{"question": "..."}` | 预检：难度/预算/知识 |
| `/api/govern` | POST | `{"question": "...", "thinking": "...", "steps_done": 0}` | 治理：stop/continue 建议 |
| `/api/audit` | POST | `{"answer": "...", "question": "..."}` | 审计：因果主张校验 |
| `/api/query` | POST | `{"mode": "effects\|causes\|path", ...}` | 因果知识查询 |
| `/api/chat` | POST | `{"text": "..."}` | 对话式因果问答（旧接口） |
| `/api/state` | GET | - | 运行状态 |

### 3. CLI

```
/precheck <问题>   # 推理预检（难度/预算/知识预取）
/audit <文本>      # 审计文本中的因果主张
/query 类          # 直接对话问 "X会怎样 / 为什么Y" 等
python main.py --selftest   # 全能力自检
```

## GGUF 大脑权重包

神经组件（预算控制器 + 收敛检测器 + 链接预测器）可打包为标准 GGUF 文件：

```bash
python scripts/export_gguf.py --train-link   # 产出 ultron_brain.gguf
```

* 写入器为**零依赖实现**，遵循 GGUF v3 规范：**每块张量数据独立按 32 字节
  对齐**，并在元数据里显式声明 `general.alignment`（2026-10 修复；旧版让张量
  首尾相接，`b2` 这类 12 字节张量之后全部错位，按规范逐张量补位的第三方读取器
  会整段读成乱码）。
* 层结构元数据由**真实张量形状反推**，不再写死常量（旧版 `[12,32,16,3]` 与
  13 维特征不符）。
* 导出脚本自带两道自检：`gguf_io.check_file()` 校验对齐/越界/重叠/元数据维度，
  以及**与 `ultron/weights/*.npz` 的逐张量一致性**（旧版没有这道护栏，曾出现
  `link_predictor.npz` 在导出 5 分钟后被重训、GGUF 里留下上一代权重的情况）。
* GGUF 是"权重容器"：因果知识图（`causal_graph.json`）与推理代码属"数据 +
  程序"，不在其中——它们以 MCP/HTTP 服务形态与大模型协同。
* 权重包内含全部超参与元数据（层结构、职责说明、图规模统计），
  可被自有工具链（`ultron/gguf_io.py`）直接读回装配。
* 也可以用官方 `gguf` 包交叉验证（可选）：`pip install gguf` 后重跑导出脚本。

> **发布建议**：`ultron_brain.gguf` 是构建产物，几秒即可重生成，不要提交进版本库
> （见 `.gitignore`），需要时挂 Release 附件。

### 神经组件的嵌入口径（重要）

`NodeEmbeddingIndex` 按嵌入维度分开缓存（`*.emb_dim{N}.npz`）：

| 缓存 | 来源 | 用途 |
|------|------|------|
| `emb_dim384.npz` | 真实语义模型 `paraphrase-multilingual-MiniLM-L12-v2` | **训练/推理都用这份** |
| `emb_dim128.npz` | 语义模型不可用时的**字符 n-gram 哈希降级** | 仅离线兜底，**不是语义向量** |

2026-10 复核发现：仓库里那份链接预测器是拿 **128 维哈希降级特征**训练的
（正负样本 AUC 0.756），而真正的 384 维语义嵌入一直没被用上。现已改为用 384 维
语义嵌入训练（`--train-link` 会在维度不一致时**自动重训**，不再静默沿用旧权重）：

```
384 维语义嵌入训练：loss=0.4244  AUC=0.883   （正 20000 / 负 40000）
```


## 思考治理对比实验

对真实推理模型（Ollama qwen3:1.7b）实测"裸跑 vs 治理"：

```bash
ollama pull qwen3:1.7b
python scripts/demo_governance.py
```

输出每个问题的：思考字符数（≈token）、耗时、答案、关键节点命中，
以及总结的思考开销节省比例。

### 实测结果（qwen3:1.7b，4 个问题）

| 问题 | 裸跑思考 | 治理后思考 | 判定 | 耗时（裸跑→治理） | 关键节点命中 |
|------|---------|-----------|------|------------------|-------------|
| 你好，请用一句话介绍你自己 | 138 tok | **0 tok** | 0 免思考（直答） | 1.4s → 0.3s | 0/4 → 0/4 |
| 熬夜会怎样？ | 276 tok | 55 tok（超预算截断） | 1 轻思考 | 6.4s → 1.2s | 1/7 → 1/7 |
| 为什么地面湿？ | 241 tok | 55 tok（超预算截断） | 1 轻思考 | 5.4s → 1.1s | 1/2 → 1/2 |
| 为什么下雨会导致迟到？ | 394 tok | 157 tok（自然收敛） | 2 深思考 | 5.7s → 2.3s | 2/14 → 2/14 |
| **合计** | **1049 tok / 18.9s** | **267 tok / 4.9s** | — | **↓74%** | **命中率持平** |

**思考开销节省 75%，端到端耗时下降 74%，答案的关键因果节点命中数不变**——
省掉的是冗余思考段，不是推理能力。这才是"别的 AI 高效推理"的可交付证据。

### 这个 75% 该怎么读（2026-10 独立复核口径）

* 三次治理截断**全部由规则分支的"预算耗尽"**触发，神经收敛检测器 0 次触发；
* 真正卡住的是 `max_steps`（思考段数），不是 `max_tokens`——难度 1 名义
  "256 tokens"，实际约 55 tok 就收束了；
* 40+ 题实测：融合后的难度与纯规则难度相比，神经侧的改动量约 5%~11%
  （即大头仍是规则 + 硬预算贡献，神经网络是纠偏层，不是主力）。

## 数据质量与许可（请连同上面的能力一起读）

### 知识图 `causal_graph.json`（53,204 节点 / 114,951 边）

| 项 | 实测 |
|---|---|
| 结构 | 0 孤立节点、0 重边、0 悬挂边、0 自环 ✅ |
| 边权重 | **82.1% 的边权重恰好 = 0.45**（ConceptNet 单来源边 raw=1.0 经对数映射后同值）→ 权重作为"可信度"几乎没有区分度 |
| 连通性 | **66.3% 的节点只有 1 条边**，49.4% 是出度为 0 的汇点（度中位数 = 1） |
| 关系类型 | `/r/Causes` 66,087、`/r/HasSubevent` 33,364、`/r/CausesDesire` 15,390 —— **后两者是"包含子事件/引发欲望"，不是因果关系**，合计占 43%，按需过滤 |
| 噪声节点 | 974 个节点含空格、757 个含拉丁字母（如 `上 ptt`、`如 radiohead`），约 1.8% |
| 体积 | 23.8MB JSON → gzip 后 **1.4MB**（要发布建议发 `.json.gz`） |

### 许可

* **代码**：[Apache License 2.0](LICENSE)（含专利授权条款）。
* **`causal_graph.json`**：派生自 **ConceptNet 5.7**，其许可为 **CC BY-SA 4.0**
  （署名 + 相同方式共享）——**不适用**本仓库的 Apache-2.0，需保留来源与派生关系：
  > 本文件由 ConceptNet 5.7（https://conceptnet.io ，CC BY-SA 4.0）中文断言
  > 经 `scripts/rebuild_graph.py` 抽取因果关系与对数权重映射派生而来，
  > 派生部分同样以 CC BY-SA 4.0 提供。
* **`data/conceptnet-assertions-5.7.0.csv.gz`**：第三方原始数据（约 475MB，
  超 GitHub 单文件 100MB 硬限），不进版本库，按需用 `scripts/rebuild_graph.py` 重建。

## 全量验收

```bash
python main.py --selftest                          # 7 大能力 + 协处理器 8 项断言
python scripts/check_interfaces.py --port 8611     # HTTP 全端点 + MCP 全工具回归
```

* `--selftest` 末段对 PreCheck / Governor / Audit 做确定性断言（难度分级、
  判停、超预算截停、主张 supported/unknown），不依赖任何外部大模型。
* `check_interfaces.py` 以**真实子进程 + 真实网络/管道**验证两条协同通道，
  并含只读模式与 `--allow-write` 写路径的隔离验证（临时图，测完即删）。

## 架构与目录

```
ultron/
├── coprocessor.py      推理协处理器：PreCheck / Governor / Audit（核心）
├── controller_nn.py    思考控制器：预算控制器 + 收敛检测器（numpy MLP）
├── synth_data.py       控制器训练数据合成（任务模拟式自监督）
├── gguf_io.py          GGUF 权重容器：零依赖写入/读取 + 大脑导出/导入
├── mcp_server.py       MCP stdio 服务（5 工具，零依赖协议实现）
├── server.py           HTTP REST API（纯接口，无前端页面）
├── bot.py              对话层（CLI 交互 + 教学状态机）
├── graph.py            因果图（符号层：条件因果/干预/反事实 + 来源元数据）
├── embeddings.py       语义嵌入（SentenceTransformer / 哈希降级）
├── link_predictor.py   链接预测器（自监督：已有边 vs 随机非边）
├── reasoner.py         思维链 / 规划器 / 蒙特卡洛 test-time compute
├── agent.py            工具+RAG+反思回路（NeuroSymbolicAgent）
├── nn.py               纯 numpy MLP 引擎（零依赖）
├── rl.py / world_model.py   教学演示组件（自我对弈 / 网格世界）
└── weights/            训练产物（控制器 / 链接预测器 .npz）

scripts/
├── rebuild_graph.py       从 ConceptNet 重建因果图（来源标注+对数权重）
├── train_controllers.py   训练思考控制器
├── export_gguf.py         导出 GGUF 大脑权重包（含官方工具交叉验证）
├── check_interfaces.py    接口回归：HTTP 全端点 + MCP 全工具
├── demo_governance.py     思考治理对比实验（Ollama 联调）
└── repo_guard.py          发布守卫：拦截本机私有内容进库

.githooks/                 提交/推送前钩子（启用方式见「发布防护」）

causal_graph.json          因果知识图（5.3 万节点 / 11.5 万边）
backups/                   图重建前的历史备份
```

```
外部大模型 (Ollama / 云端 API / Agent 框架)
   │  MCP · HTTP · CLI
   ▼
Ultron 推理协处理器
   ├─ PreCheck  难度 · 预算 · 知识预取
   ├─ Governor  收敛检测 · 循环识别 · 截停
   ├─ Audit     因果一致性 · 引用校验
   └─ 内核      因果图 5.3 万节点 + 神经控制器（可导出 GGUF）
```

## 发布防护

本仓库带一道**本地守卫**，在内容离开本机前拦截三类风险：私有/可再生目录
（`.workbuddy/`、`data/`、`backups/`）、密钥与大二进制（`.npz`/`.gguf`/`.pem`…，
仅 `causal_graph.json` 与 `ultron/weights/*.npz` 白名单放行）、以及正文中的
本机绝对路径、系统用户名、真实邮箱、访问令牌和私钥头。

```bash
git config core.hooksPath .githooks     # 一次性启用钩子（每个克隆都要做）
python scripts/repo_guard.py --all      # 手工体检：全库已跟踪文件
```

钩子默认不随克隆自动生效——`core.hooksPath` 是本地配置。没启用时，
`--all` 仍可随时做一次全库扫描。确需提交被拦内容（已人工核实）时，
用 `ULTRON_GUARD_ALLOW=1 git commit ...` 临时放行。

## 环境变量

| 变量 | 说明 |
|------|------|
| `HF_ENDPOINT` | 嵌入模型下载镜像（默认 `https://hf-mirror.com`） |
| `ULTRON_GUARD_ALLOW` | 设为 `1` 时跳过 `repo_guard.py` 检查（已人工核实后使用） |
| `GIT_SSL_NO_VERIFY` | 仅在企业 TLS 中间人代理环境下推送时需要设为 `true` |
