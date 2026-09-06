"""多格式文档解析：PDF / Word(docx) / TXT / Markdown → 统一的结构化 Block 列表。

设计要点：解析结果不是纯文本，而是带结构的 Block：
- kind:  heading | paragraph | table | list
- page:  PDF 页码（其余格式为 None）
- level: 标题层级（1 起），非标题为 0
后续的智能切分依赖这些结构信息还原「文档 > 章节」层级，实现段落级溯源。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

***REMOVED*** ---------- 数据结构 ----------


@dataclass
class Block:
    text: str
    kind: str = "paragraph"  ***REMOVED*** heading | paragraph | table | list
    page: int | None = None
    level: int = 0


@dataclass
class ParsedDoc:
    doc_name: str
    source_path: str
    blocks: list[Block] = field(default_factory=list)


***REMOVED*** ---------- 标题识别 ----------

***REMOVED*** 强特征：几乎可以断定是标题
_STRONG_HEADINGS = [
    re.compile(r"^第\s*[一二三四五六七八九十百0-9]+\s*[章节篇讲]"),
    re.compile(r"^***REMOVED***{1,6}\s+"),                       ***REMOVED*** markdown
    re.compile(r"^Q\s*\d+\s*[：:.?？]"),              ***REMOVED*** FAQ 问答体
    re.compile(r"^\d{1,2}(\.\d{1,2}){1,3}[\s、.．]\s*\S"),  ***REMOVED*** 3.1 / 6.2.1 需跟分隔符
]
***REMOVED*** 弱特征：短行才可信
_WEAK_HEADINGS = [
    re.compile(r"^[一二三四五六七八九十]+\s*、"),
    re.compile(r"^[（(][一二三四五六七八九十0-9]+[）)]"),
    re.compile(r"^\d{1,2}[\s、.．]\s*\S+"),
]
_BLACKLIST_TAIL = ("。", "，", "！", "？", "；", ";", ",", "：", ":")


def is_heading_text(text: str) -> bool:
    """根据文本形态判断是否为标题（用于无样式信息的 TXT/弱结构 PDF）。"""
    text = text.strip()
    if not text or len(text) > 40:
        return False
    if any(p.match(text) for p in _STRONG_HEADINGS):
        return True  ***REMOVED*** 强特征（Q1：/第四章/3.1/***REMOVED***）即使以问号结尾也是标题
    if text.endswith(_BLACKLIST_TAIL):
        return False
    return len(text) <= 30 and any(p.match(text) for p in _WEAK_HEADINGS)


def _clean_heading(text: str) -> str:
    return text.lstrip("***REMOVED***").strip()


***REMOVED*** ---------- PDF ----------


def parse_pdf(path: Path) -> ParsedDoc:
    """按页解析 PDF。

    标题判定优先用字体大小（行内最大字号显著大于正文中位数），
    辅以文本形态正则；同一视觉块内的行合并为一段，
    跨视觉块但语义连续的段落（前块无句末标点）再合并。
    """
    import pymupdf  ***REMOVED*** PyMuPDF

    pd = ParsedDoc(path.name, str(path))
    pdf = pymupdf.open(path)
    try:
        for pno, page in enumerate(pdf, start=1):
            data = page.get_text("dict")
            sizes = [
                span["size"]
                for block in data["blocks"]
                if block.get("type") == 0
                for line in block["lines"]
                for span in line["spans"]
                if span["text"].strip()
            ]
            body_size = _median(sizes) if sizes else 10.5

            for block in data["blocks"]:
                if block.get("type") != 0:
                    continue
                lines = []
                for line in block["lines"]:
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if text:
                        lines.append((text, max(s["size"] for s in line["spans"] if s["text"].strip())))
                if not lines:
                    continue

                ***REMOVED*** 行 → 段：标题行单独成块；普通行合并（中文直接拼接，英文补空格）
                buf_text, buf_size = "", 0.0
                for text, size in lines:
                    if _looks_like_heading(text, size, body_size):
                        if buf_text:
                            pd.blocks.append(Block(buf_text, "paragraph", pno))
                            buf_text, buf_size = "", 0.0
                        pd.blocks.append(Block(_clean_heading(text), "heading", pno,
                                               1 if size >= body_size * 1.35 else 2))
                    elif is_heading_text(text) and len(text) <= 30:
                        if buf_text:
                            pd.blocks.append(Block(buf_text, "paragraph", pno))
                            buf_text, buf_size = "", 0.0
                        pd.blocks.append(Block(_clean_heading(text), "heading", pno, 1))
                    else:
                        buf_text = _smart_join(buf_text, text)
                        buf_size = max(buf_size, size)
                if buf_text:
                    pd.blocks.append(Block(buf_text, "paragraph", pno))
    finally:
        pdf.close()

    ***REMOVED*** 后处理：合并被 MuPDF 拆开的连续段落块（前块无句末标点 → 语义连续）
    merged: list[Block] = []
    for b in pd.blocks:
        if (merged and merged[-1].kind == "paragraph" and b.kind == "paragraph"
                and merged[-1].page == b.page and merged[-1].text
                and not merged[-1].text.endswith(tuple("。！？；：.!?"))):
            merged[-1].text = _smart_join(merged[-1].text, b.text)
        else:
            merged.append(b)
    pd.blocks = merged
    return pd


def _looks_like_heading(text: str, size: float, body_size: float) -> bool:
    return len(text) <= 40 and size >= body_size * 1.15 and not text.endswith(_BLACKLIST_TAIL)


def _median(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    if n == 0:
        return 10.5
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def _smart_join(a: str, b: str) -> str:
    if not a:
        return b
    cjk = lambda ch: "\u4e00" <= ch <= "\u9fff"  ***REMOVED*** noqa: E731
    sep = "" if (cjk(a[-1]) or cjk(b[:1]) or b[:1] in "，。、；：！？（）") else " "
    return a + sep + b


***REMOVED*** ---------- Word (.docx) ----------


def parse_docx(path: Path) -> ParsedDoc:
    """解析 Word：按 body 顺序遍历段落与表格，保留标题样式层级。"""
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    pd = ParsedDoc(path.name, str(path))
    d = docx.Document(str(path))
    for child in d.element.body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, d)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            m = re.search(r"heading (\d)|标题 (\d)", style)
            if m:
                pd.blocks.append(Block(text, "heading", None, int(m.group(1) or m.group(2))))
            elif is_heading_text(text):
                pd.blocks.append(Block(text, "heading", None, 1))
            else:
                pd.blocks.append(Block(text, "paragraph", None))
        elif child.tag == qn("w:tbl"):
            table = Table(child, d)
            rows = []
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                pd.blocks.append(Block("\n".join(rows), "table", None))
    return pd


***REMOVED*** ---------- TXT / Markdown ----------


def parse_txt(path: Path) -> ParsedDoc:
    """解析纯文本/Markdown：正则识别章节标题，空行分段。"""
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8", "gb18030", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"无法识别文件编码: {path}")

    pd = ParsedDoc(path.name, str(path))
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        first = lines[0]
        if first.startswith("***REMOVED***"):
            level = max(1, len(first) - len(first.lstrip("***REMOVED***")))
            pd.blocks.append(Block(_clean_heading(first), "heading", None, level))
            rest = "\n".join(lines[1:]).strip()
            if rest:
                pd.blocks.append(Block(rest, "paragraph", None))
        elif is_heading_text(first) and len(lines) <= 3 and len(first) <= 35:
            pd.blocks.append(Block(first, "heading", None, 1))
            rest = "\n".join(lines[1:]).strip()
            if rest:
                pd.blocks.append(Block(rest, "paragraph", None))
        else:
            pd.blocks.append(Block("\n".join(lines), "paragraph", None))
    return pd


***REMOVED*** ---------- 统一入口 ----------

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}


def parse_file(path: str | Path) -> ParsedDoc:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext in (".txt", ".md"):
        return parse_txt(path)
    raise ValueError(f"暂不支持的格式: {ext}（支持 {sorted(SUPPORTED_EXTS)}）")
