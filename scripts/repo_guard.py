"""仓库守卫：阻止本机私有内容被误推到公开仓库。

三道检查（全部在**提交前/推送前**本地执行，联网与否都能跑）：
  1. 路径黑名单：本机工作区状态（.workbuddy/）、原始数据（data/）、备份、
     虚拟环境、缓存目录一律不许进库；
  2. 类型黑名单：密钥、大二进制、可再生的缓存产物（.npz/.gguf/.pem/.key…），
     仅对显式白名单（causal_graph.json、ultron/weights/*.npz）放行；
  3. 内容扫描：本机绝对路径、系统用户名、真实邮箱、访问令牌、私钥头。
     邮箱放行 `*@users.noreply.github.com`（GitHub 隐私邮箱）。

用法：
    python scripts/repo_guard.py --staged        # pre-commit：查暂存区
    python scripts/repo_guard.py --push          # pre-push：查待推送提交
    python scripts/repo_guard.py --all           # 手工体检：查全库已跟踪文件

退出码：0 通过；1 发现风险（已拦截）。

临时放行（确需提交被拦内容时）：
    ULTRON_GUARD_ALLOW=1 git commit ...
"""

import os
import re
import subprocess
import sys
from typing import List, Set, Tuple

# ==================== 规则 ====================

# 路径前缀黑名单：这些目录属于本机私有/可再生产物
DENY_PREFIX = (
    ".workbuddy/", ".git/", "data/", "backups/",
    ".venv/", "venv/", "env/", "__pycache__/", ".idea/", ".vscode/",
)

# 扩展名黑名单：密钥、二进制、可再生缓存
DENY_EXT = {
    ".npz", ".gguf", ".pem", ".key", ".p12", ".pfx", ".crt", ".der",
    ".log", ".sqlite", ".db", ".zip", ".7z", ".rar", ".tar", ".gz",
    ".pdf", ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".ico", ".mp4", ".mp3", ".wav", ".bin", ".exe", ".dll",
}

# 文件名黑名单
DENY_NAME = {
    ".env", "id_rsa", "id_ed25519", "id_ecdsa", "credentials", "auth.json",
    ".netrc", ".npmrc", ".pypirc",
}

# 显式白名单：确定属于项目内容、体积大但必须入库的文件
ALLOW_EXACT = {
    "causal_graph.json",        # 22.7 MB，开箱即用的知识图
    "causal_graph.preseed.json",
}
ALLOW_PREFIX = ("ultron/weights/",)   # 训练产物，1.8 MB 以内

# 单个文件体积上限（白名单之外）
MAX_BYTES = 10 * 1024 * 1024

# 内容扫描规则：(说明, 正则)
CONTENT_RULES: List[Tuple[str, "re.Pattern"]] = [
    ("本机绝对路径", re.compile(r"[A-Za-z]:[\\/](?:Users|work)[\\/]", re.I)),
    ("系统用户名", re.compile(r"\baqinh\b", re.I)),
    ("私钥文件头", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("GitHub 令牌", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}")),
    ("GitHub 细粒度令牌", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("OpenAI/Anthropic 风格密钥", re.compile(r"\b(?:sk|sk-ant)-[A-Za-z0-9_-]{16,}")),
    ("Slack 令牌", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("HTTP Bearer 令牌", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}")),
]

# 允许出现的邮箱（GitHub 隐私邮箱 / 示例域名）
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
EMAIL_ALLOW_DOMAIN = ("users.noreply.github.com", "example.com", "example.org")


# ==================== git 工具 ====================

def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace").strip())
    return out.stdout.decode("utf-8", "replace")


def _run(*args: str) -> Tuple[int, bytes]:
    p = subprocess.run(["git", *args], capture_output=True)
    return p.returncode, p.stdout


def staged_files() -> List[str]:
    """暂存区新增/修改的文件（不含删除）。"""
    _, raw = _run("diff", "--cached", "--name-only", "--diff-filter=ACM")
    return [x for x in raw.decode("utf-8", "replace").splitlines() if x]


def push_files() -> List[str]:
    """待推送提交涉及的文件。"""
    files: Set[str] = set()
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        local_sha, remote_sha = parts[1], parts[3]
        zero = set(remote_sha)
        rng = (f"{remote_sha}..{local_sha}" if zero != {"0"}
               else f"{local_sha}")          # 新分支：整条历史
        _, raw = _run("log", "--pretty=format:", "--name-only", rng)
        files.update(x for x in raw.decode("utf-8", "replace").splitlines() if x)
    return sorted(files)


def tracked_files() -> List[str]:
    _, raw = _run("ls-files")
    return [x for x in raw.decode("utf-8", "replace").splitlines() if x]


def blob(path: str, staged: bool) -> bytes:
    spec = f":{path}" if staged else f"HEAD:{path}"
    code, data = _run("cat-file", "blob", spec)
    if code != 0:                      # 回退到工作区
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return b""
    return data


# ==================== 检查 ====================

def _is_allowed(path: str) -> bool:
    return path in ALLOW_EXACT or path.startswith(ALLOW_PREFIX)


def check_path(path: str) -> List[str]:
    """返回该路径命中的问题列表（空表示通过）。"""
    problems: List[str] = []
    low = path.lower()
    if low.startswith(DENY_PREFIX):
        problems.append(f"位于私有/可再生目录（{low.split('/')[0]}/）")
    name = os.path.basename(low)
    if name in DENY_NAME:
        problems.append("疑似凭据文件（按文件名）")
    ext = os.path.splitext(low)[1]
    if ext in DENY_EXT and not _is_allowed(path):
        problems.append(f"受限文件类型 {ext}（密钥/二进制/可再生缓存）")
    return problems


def check_content(path: str, data: bytes) -> List[str]:
    """扫描文本内容中的隐私与凭据特征。"""
    problems: List[str] = []
    if b"\x00" in data[:4096]:         # 二进制跳过内容扫描
        return problems
    text = data.decode("utf-8", "replace")
    for label, rx in CONTENT_RULES:
        m = rx.search(text)
        if m:
            problems.append(f"包含{label}：{m.group(0)[:24]!r}")
    for m in EMAIL_RE.finditer(text):
        if not m.group(1).lower().endswith(EMAIL_ALLOW_DOMAIN):
            problems.append(f"包含真实邮箱：{m.group(0)!r}")
            break
    return problems


def scan(files: List[str], staged: bool, check_size: bool) -> List[Tuple[str, List[str]]]:
    findings: List[Tuple[str, List[str]]] = []
    for path in files:
        problems = check_path(path)
        data = blob(path, staged)
        if check_size and len(data) > MAX_BYTES and not _is_allowed(path):
            problems.append(f"体积 {len(data) / 1024 / 1024:.1f} MB 超过 10 MB 上限")
        problems += check_content(path, data)
        if problems:
            findings.append((path, problems))
    return findings


def report(findings: List[Tuple[str, List[str]]], scope: str) -> int:
    if not findings:
        print(f"[守卫] {scope}检查通过：未发现本机私有内容。")
        return 0
    print("=" * 66)
    print(f"[守卫] 已拦截：{scope}发现 {len(findings)} 个文件可能泄露本机内容")
    print("=" * 66)
    for path, problems in findings:
        print(f"  ✗ {path}")
        for p in problems:
            print(f"      - {p}")
    print("-" * 66)
    print("处理建议：")
    print("  · 私有/可再生内容：确认 .gitignore 已覆盖，或 git rm --cached <文件>")
    print("  · 内容含本机路径/凭据：改从环境变量或配置读取，切勿硬编码")
    print("  · 确需提交（已人工核实）：ULTRON_GUARD_ALLOW=1 git commit ...")
    print("=" * 66)
    return 1


def main(argv: List[str]) -> int:
    if os.environ.get("ULTRON_GUARD_ALLOW") == "1":
        print("[守卫] 检测到 ULTRON_GUARD_ALLOW=1，本次跳过检查。")
        return 0
    if "--push" in argv:
        return report(scan(push_files(), staged=False, check_size=True),
                      "待推送提交")
    if "--all" in argv:
        return report(scan(tracked_files(), staged=False, check_size=True),
                      "全库已跟踪文件")
    if "--staged" in argv:
        return report(scan(staged_files(), staged=True, check_size=False),
                      "暂存区")
    print("用法：repo_guard.py --staged | --push | --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
