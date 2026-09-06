"""安全加固：输入过滤、提示注入检测、引用校验、审计/拦截日志。"""

import json
import re
import time
from pathlib import Path

from .config import ROOT

LOG_DIR = ROOT / "logs"
BLOCKED_LOG = LOG_DIR / "blocked_queries.log"
AUDIT_LOG = LOG_DIR / "audit.jsonl"

MAX_QUESTION_CHARS = 500
***REMOVED*** 零宽字符与控制字符（保留换行）
_CONTROL_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\t\x00-\x08\x0b\x0c\x0e-\x1f]")

***REMOVED*** 提示注入特征（大小写不敏感）；命中即拦截。覆盖指令覆盖/提示词套取/角色劫持/越狱四大类。
_INJECTION_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"忽略(之前|以上|上述|上面|前面|先前)(的)?(所有|全部)?(指令|提示|规则|要求|设定)",
        r"ignore (all |any )?(previous|prior|above|earlier) (instructions|prompts|rules)",
        r"disregard (your|all|the) (instructions|guidelines)",
        r"(系统|system)\s*(提示词|prompt)",
        r"(泄露|输出|打印|告诉我|复述).{0,6}(提示词|system prompt|初始指令|系统设定)",
        r"你(现在)?是(一个)?(新的|另一个|不同的).{0,8}(助手|AI|模型|角色)",
        r"进入(开发者|调试|维护|god|dev)模式",
        r"(忘记|无视|无视掉)(之前|以上|上述)(的)?(所有)?(指令|设定|规则)",
        r"break (out of|the) (sandbox|rules)",
        r"jailbreak|DAN模式|越狱",
        r"现在开始(不用|无需)(遵守|理会)(任何)?(规则|限制)",
        r"请?扮演(一个)?(不受|没有)(道德|法律|规则)(限制|约束)的",
        r"repeat (everything|your (system )?prompt)",
        r"输出你(的)?(初始|系统)消息",
    ]
]

***REMOVED*** 拒答类回答的识别（引用校验时用于区分"拒答"与"实质性回答"）
_REFUSAL_RE = re.compile(r"(未找到|没有找到|无法回答|未提及|没有相关|找不到|无法回答这个问题|i don't know|not found)", re.I)


class InputBlocked(Exception):
    """用户输入被安全策略拦截。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def sanitize_input(text: str) -> str:
    """清洗输入：去零宽/控制字符、截断超长。"""
    text = _CONTROL_RE.sub("", text or "")
    return text.strip()[:MAX_QUESTION_CHARS]


def check_input(text: str, log: bool = True) -> str:
    """返回清洗后的输入；命中注入特征时抛 InputBlocked 并记录拦截日志。"""
    cleaned = sanitize_input(text)
    if not cleaned:
        raise InputBlocked("输入为空")
    for pat in _INJECTION_PATTERNS:
        m = pat.search(cleaned)
        if m:
            if log:
                _append_log(BLOCKED_LOG, {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": f"提示注入特征: {m.group(0)!r}",
                    "input": cleaned[:200],
                })
            raise InputBlocked(f"输入包含不被允许的指令内容（命中规则：{m.group(0)[:20]}…）")
    return cleaned


def validate_citations(answer: str, n_chunks: int) -> tuple[list[int], list[int]]:
    """校验答案中的引用编号。返回 (有效编号, 伪造编号)。

    伪造 = 超出检索结果范围（如只有 5 个块却出现 [9]）或非正整数。
    """
    nums = [int(n) for n in re.findall(r"\[(\d{1,3})\]", answer)]
    valid = sorted({n for n in nums if 1 <= n <= n_chunks})
    forged = sorted({n for n in nums if n < 1 or n > n_chunks})
    return valid, forged


def needs_citation(answer: str) -> bool:
    """实质性回答（非拒答）应当带引用。"""
    return len(answer) >= 30 and not _REFUSAL_RE.search(answer)


def _append_log(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  ***REMOVED*** 日志失败不影响主流程


def audit(event: dict) -> None:
    """问答审计日志：完整 prompt、检索结果、答案与引用映射。"""
    event = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    _append_log(AUDIT_LOG, event)
