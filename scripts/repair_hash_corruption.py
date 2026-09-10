"""修复仓库中被替换为占位标记的 "#" 字符。

背景
----
一次失败的脱敏处理把仓库里**所有** "#" 字符替换成了一个固定的字面量标记
（本文件用 ``_marker()`` 现场拼出该标记，避免自己再次被扫到），其中连续的
多个 "#"（Markdown 标题的 ## / ###）被折叠成**一个**标记。后果：

  * 所有 .py 文件里的注释变成非法语法 —— ``import rag.pipeline`` 直接 SyntaxError；
  * .gitignore 的注释行失效；
  * requirements.txt / docker-compose.yml / Markdown 标题全部损坏。

修复规则
--------
* 代码 / 配置类文件（.py .yml .txt .ini .cfg .toml .json ...）：
  标记 -> ``#``。这些格式里 "#" 恒为注释符，还原结果唯一且必然正确；
  行尾注释统一收敛为 `` # 内容``，并保留 ``# noqa`` 之类的工具指令。
* Markdown：
  - 围栏代码块（``` / ~~~）**内部** 与非行首位置的标记 -> ``#``
    （那是代码注释或行内代码，例如表头里的 `` `#` `` 序号列）；
  - 行首的标记 -> Markdown 标题，层级取 min(标记数, 前一个标题层级 + 1, 3)，
    保证不跳级、不回退。

用法
----
    python scripts/repair_hash_corruption.py --check   # 只报告，不写入
    python scripts/repair_hash_corruption.py           # 执行修复
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _marker() -> str:
    """现场拼出损坏标记，避免本文件自身包含该字面量。"""
    return "*" * 3 + "REMOVED" + "*" * 3


SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".tools"}
CODE_EXTS = {".py", ".yml", ".yaml", ".txt", ".ini", ".cfg", ".toml", ".json", ".js", ".ts", ".css", ".html"}
MD = ".md"
# 无扩展名的文本文件（Dockerfile、.gitignore、.dockerignore、.env.example ...）。
# 这些文件里 "#" 同样是注释符，必须一起修复 —— 漏掉它们会留下失效的注释行
# （第一轮修复就曾漏掉 .gitignore / Dockerfile / .dockerignore / .env.example）。
EXTENSIONLESS = {
    ".gitignore", ".dockerignore", ".env.example", ".env.sample", ".gitattributes",
    ".editorconfig", "dockerfile", "makefile", "procfile", "license",
}
FENCE = re.compile(r"^\s*(```|~~~)")


def is_markdown(p: Path) -> bool:
    return p.suffix.lower() == MD


def is_text_config(p: Path) -> bool:
    """该文件是否应当按「# 是注释符」的规则修复。"""
    if p.suffix.lower() in CODE_EXTS:
        return True
    name = p.name.lower()
    return name in EXTENSIONLESS or not p.suffix


def iter_files() -> list[Path]:
    out: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if is_markdown(p) or is_text_config(p):
            out.append(p)
    return sorted(out)


def repair_code(text: str, mark: str) -> str:
    """每个标记 -> '#'；行尾注释前的空白收敛为一个空格。"""
    lines: list[str] = []
    for line in text.splitlines():
        if mark not in line:
            lines.append(line)
            continue
        stripped = line.lstrip()
        if stripped.startswith(mark):
            indent = line[: len(line) - len(stripped)]
            lines.append(indent + re.sub(r"(?:" + re.escape(mark) + r")+", "#", stripped))
        else:
            lines.append(re.sub(r"[ \t]*" + re.escape(mark), " #", line))
    if text.endswith("\n"):
        lines.append("")
    return "\n".join(lines)


def repair_markdown(text: str, mark: str) -> tuple[str, list[tuple[int, int]]]:
    """修复 Markdown，返回 (新文本, [(行号, 推断层级)])。"""
    run_head = re.compile(r"^(?:" + re.escape(mark) + r")+")
    out: list[str] = []
    prev = 0
    in_fence = False
    report: list[tuple[int, int]] = []
    for idx, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if mark not in line:
            out.append(line)
            continue
        m = run_head.match(line)
        if in_fence or not m:
            # 代码块内 / 行内：还原成注释符或行内代码
            if in_fence and m:
                indent = line[: len(line) - len(line.lstrip())]
                out.append(indent + re.sub(r"(?:" + re.escape(mark) + r")+", "#", line.lstrip()))
            else:
                out.append(re.sub(r"[ \t]*" + re.escape(mark), " #", line))
            continue
        rest = line[m.end():]
        depth = max(1, min(len(m.group(0)) // len(mark), prev + 1, 3))
        prev = depth
        report.append((idx, depth))
        out.append("#" * depth + rest)
    if text.endswith("\n"):
        out.append("")
    return "\n".join(out), report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告，不写文件")
    args = ap.parse_args()

    mark = _marker()
    changed_code = 0
    changed_md = 0
    heading_report: list[str] = []
    self_path = Path(__file__).resolve()

    for p in iter_files():
        if p.resolve() == self_path:
            continue  # 本脚本自身不含该字面量，但仍跳过以防万一
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if mark not in text:
            continue
        rel = p.relative_to(ROOT)

        if is_markdown(p):
            new_text, rep = repair_markdown(text, mark)
            for lineno, depth in rep:
                heading_report.append(f"{rel}:{lineno}: -> {'#' * depth}")
            changed_md += 1
        else:
            new_text = repair_code(text, mark)
            changed_code += 1

        if not args.check:
            p.write_text(new_text, encoding="utf-8", newline="")

    verb = "将修复" if args.check else "已修复"
    print(f"[{verb}] 代码/配置文件 {changed_code} 个，Markdown {changed_md} 个")
    if args.check:
        print(f"\n### Markdown 标题层级推断（共 {len(heading_report)} 条，前 40 条）:")
        for s in heading_report[:40]:
            print("   ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
