# -*- coding: utf-8 -*-
"""提交/历史脱敏审计：扫描密钥、私有端点与个人敏感信息。

用法：
    python scripts/audit_secrets.py              # 扫描全部历史（默认）
    python scripts/audit_secrets.py --head       # 只扫当前工作树（快，适合提交前）
    python scripts/audit_secrets.py --staged     # 只扫已暂存内容（pre-commit 用）

退出码：0 = 通过；1 = 发现疑似泄露（CI 会因此失败）。

为什么默认扫全部历史：
    把敏感值从最新代码里删掉，并不能让它从 git 历史里消失——clone 之后
    `git log -p` 依然能翻出旧提交里的明文。本仓库就曾把私有推理端点写进过
    早期提交，因此审计必须覆盖历史。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict

# Windows 控制台默认 GBK，直接 print emoji/特殊符号会抛 UnicodeEncodeError。
# 统一切到 UTF-8 并对无法编码的字符降级替换，保证脚本在任何终端都能跑完。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 — 老版本解释器无 reconfigure
    pass

# 高置信度特征。宁可多报，由 ALLOW 白名单收敛误报。
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("API Key (sk-)", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("Bearer Token", re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.I)),
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub Token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("私钥文件内容", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("URL 内嵌密码", re.compile(r"://[^/\s:@]{3,}:[^/\s:@]{3,}@")),
    ("私有推理端点", re.compile(r"ws-[a-z0-9]{10,}\.[a-z0-9\-]*\.?(?:maas\.)?aliyuncs\.com")),
    # 内网 IP：用 \b 而不是 (?<![\d.]) 做边界。
    # 关键点有两个，都是实测踩出来的：
    #   1) 必须用 \b —— `(?<![\w.])` 会挡掉紧跟在点号后面的地址，
    #      于是 "redis://192.168.1.10" 反而检测不到；
    #   2) 每个八位组都要吃掉（(?:\.\d{1,3}){3}），否则 npm 版本号会误报：
    #      `10.13.0` 里的一段恰好长得像 10.x.x 私有网段。
    ("内网 IP", re.compile(
        r"\b(?:192\.168|172\.(?:1[6-9]|2\d|3[01])|10)(?:\.\d{1,3}){3}\b")),
    ("内网域名", re.compile(
        r"(?<![.\w])(?!threading|locale|gettext)[a-z0-9\-]{3,}\.(?:internal|corp|intranet)(?![.\w])",
        re.I)),
    # 用 (?![\d.]) 收尾：否则会命中置信区间之类的浮点数（如 0.011627906976744186）
    ("手机号", re.compile(r"(?<![\d.])1[3-9]\d{9}(?![\d.])")),
    ("身份证号", re.compile(r"(?<![\d.])\d{17}[\dXx](?![\d.])")),
    ("邮箱", re.compile(r"(?<![.\w])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?![.\w])")),
]

# 占位符 / 文档示例值：命中这些不算泄露
ALLOW = re.compile(
    r"(你的密钥|你的key|sk-xxx|sk-yyy|sk-your|placeholder|example\.com|"
    r"sk-abcdefghijklmnop1234|110101199003077777|test@example|"
    r"liming@email\.com|138\*{4}5678|@email\.com)",
    re.I,
)

# 尖括号占位符：文档模板里写 `://<db_user>:<db_password>@<db_host>` 这类形式时，
# 正则会把 "…<db_password>@" 当成"URL 内嵌密码"误报。凡是账号或口令部分
# 被尖括号包住的，一律视为占位符跳过。
PLACEHOLDER_CREDS = re.compile(r"://<[^>]*>:<[^>]*>@")


def _is_placeholder(match: str) -> bool:
    """判断命中片段是否只是文档占位符。"""
    return bool(PLACEHOLDER_CREDS.search(match))

# 二进制/大文件不扫描
SKIP_EXT = re.compile(
    r"\.(png|jpe?g|gif|webp|bmp|ico|pdf|docx?|xlsx?|pptx?|zip|gz|7z|rar|pkl|bin|exe|dll|"
    r"woff2?|ttf|otf|mp4|mp3|onnx|pt|safetensors|db|sqlite3?)$", re.I)


def _git(repo: str, args: list[str]) -> str:
    return subprocess.run(["git", "-C", repo] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def _scan_text(text: str, label: str, findings: dict) -> None:
    for line in text.splitlines():
        if SKIP_EXT.search(line[:200]):
            continue
        for name, pat in PATTERNS:
            for m in pat.finditer(line):
                val = m.group(0)
                if ALLOW.search(val) or ALLOW.search(line):
                    continue
                if _is_placeholder(val):
                    continue
                findings[name].append((label, val))


def audit_history(repo: str) -> dict:
    """逐提交扫描 diff，能定位到具体提交与文件。"""
    findings: dict = defaultdict(list)
    log = _git(repo, ["log", "--all", "--no-color", "--format=@@@%h|%s", "-p"])
    commit, path = "?", "?"
    for line in log.splitlines():
        if line.startswith("@@@"):
            commit = line[3:].strip()
        elif line.startswith("+++ b/"):
            path = line[6:].strip()
        elif line.startswith("+"):
            _scan_text(line[1:], f"{commit}  {path}", findings)
    return findings


def audit_worktree(repo: str) -> dict:
    findings: dict = defaultdict(list)
    files = [f for f in _git(repo, ["ls-files"]).splitlines() if f.strip()]
    for f in files:
        if SKIP_EXT.search(f):
            continue
        try:
            with open(f"{repo}/{f}", "r", encoding="utf-8", errors="ignore") as fh:
                _scan_text(fh.read(), f"工作树  {f}", findings)
        except OSError:
            continue
    return findings


def audit_staged(repo: str) -> dict:
    findings: dict = defaultdict(list)
    diff = _git(repo, ["diff", "--cached", "--no-color", "-U0"])
    path = "?"
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif line.startswith("+") and not line.startswith("+++"):
            _scan_text(line[1:], f"暂存区  {path}", findings)
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="仓库路径（默认当前目录）")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--head", action="store_true", help="只扫工作树")
    mode.add_argument("--staged", action="store_true", help="只扫暂存区")
    args = ap.parse_args()

    repo = args.repo
    if args.staged:
        findings = audit_staged(repo)
        scope = "暂存区"
    elif args.head:
        findings = audit_worktree(repo)
        scope = "工作树"
    else:
        findings = audit_history(repo)
        scope = "全部历史"

    print(f"[审计] 范围={scope}  仓库={repo}")
    if not findings:
        print("[通过] 未发现密钥 / 私有端点 / 个人敏感信息")
        return 0

    total = sum(len(v) for v in findings.values())
    print(f"[发现] 共 {total} 处疑似泄露：\n")
    for name, items in sorted(findings.items()):
        uniq = sorted(set(items))
        print(f"■ {name}（{len(uniq)} 处）")
        for label, val in uniq[:5]:
            masked = val if len(val) <= 16 else val[:8] + "…" + val[-4:]
            print(f"    {label}\n        {masked}")
        if len(uniq) > 5:
            print(f"    …… 另有 {len(uniq) - 5} 处")
        print()
    print("处理建议：把真实值移入 .env（已 gitignore），代码里只保留占位符；")
    print("若已进入历史，需用 git filter-repo 重写历史后再强制推送。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
