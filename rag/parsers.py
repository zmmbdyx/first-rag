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

# ---------- 数据结构 ----------


@dataclass
class Block:
    text: str
    kind: str = "paragraph" # heading | paragraph | table | list
    page: int | None = None
    level: int = 0
    rows: list[list[str]] | None = None # 表格数据（kind="table" 时存在）
    header: list[str] | None = None # 表头行（无表头为 None）


_GENERIC_HEADER = {"项目", "参数项", "名称", "配置项", "项目名称"}


def render_table(header: list[str] | None, rows: list[list[str]]) -> str:
    """把结构化表格渲染为可检索文本：每行「表头=单元格」形式，保留列语义。

    例：额定功率=1350W —— 相比裸单元格拼接，向量与 BM25 都能利用表头词，
    问答"额定功率是多少"时召回更准。两列"项目/参数"型 KV 表直接渲染为 key=value。
    """
    lines = []
    if header and any(h.strip() for h in header):
        lines.append(" | ".join(h.strip() for h in header if h.strip()))
    kv_mode = bool(header) and header[0].strip() in _GENERIC_HEADER
    for r in rows:
        if kv_mode and len(r) == 2:
            k, v = r[0].strip(), r[1].strip()
            if k or v:
                lines.append(f"{k}={v}")
            continue
        cells = []
        for i, c in enumerate(r):
            c = (c or "").strip().replace("\n", " ")
            h = header[i].strip() if header and i < len(header) and header[i] else ""
            cells.append(f"{h}={c}" if h and c else (c or h))
        line = " | ".join(x for x in cells if x)
        if line:
            lines.append(line)
    return "\n".join(lines)


@dataclass
class ParsedDoc:
    doc_name: str
    source_path: str
    blocks: list[Block] = field(default_factory=list)


# ---------- 标题识别 ----------

# 强特征：几乎可以断定是标题
_STRONG_HEADINGS = [
    re.compile(r"^第\s*[一二三四五六七八九十百0-9]+\s*[章节篇讲]"),
    re.compile(r"^ #{1,6}\s+"), # markdown
    re.compile(r"^Q\s*\d+\s*[：:.?？]"), # FAQ 问答体
    re.compile(r"^\d{1,2}(\.\d{1,2}){1,3}[\s、.．]\s*\S"), # 3.1 / 6.2.1 需跟分隔符
]
# 弱特征：短行才可信
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
        return True # 强特征（Q1：/第四章/3.1/ #）即使以问号结尾也是标题
    if text.endswith(_BLACKLIST_TAIL):
        return False
    return len(text) <= 30 and any(p.match(text) for p in _WEAK_HEADINGS)


def _clean_heading(text: str) -> str:
    return text.lstrip(" #").strip()


# ---------- PDF ----------


def parse_pdf(path: Path) -> ParsedDoc:
    """按页解析 PDF。

    标题判定优先用字体大小（行内最大字号显著大于正文中位数），辅以文本形态正则。
    表格用 PyMuPDF find_tables 结构化提取（跳过表格区域内的普通文本块），
    连续页的同列数续表自动合并并回填表头。
    """
    import pymupdf # PyMuPDF

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

            # --- 表格结构化提取 ---
            tboxes: list[tuple[float, float, float, float]] = []
            items: list[tuple[float, Block]] = [] # (y0, Block) 用于页内按版面顺序排列
            try:
                for t in page.find_tables():
                    grid = [[(c or "").replace("\n", " ").strip() for c in row] for row in t.extract()]
                    grid = [r for r in grid if any(r)]
                    if len(grid) < 2 or len(grid[0]) < 2:
                        continue
                    header = grid[0] if len(grid) > 2 and all(grid[0]) else None
                    rows = grid[1:] if header else grid
                    items.append((t.bbox[1], Block(render_table(header, rows), "table", pno,
                                                   rows=rows, header=header)))
                    tboxes.append(tuple(t.bbox))
            except Exception: # noqa: BLE001 — 个别页面表格检测失败不影响文本
                pass

            def _in_table(blk) -> bool:
                x0, y0, x1, y1 = blk["bbox"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                return any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in tboxes)

            # --- 普通文本块（行→段：标题行单独成块；普通行合并） ---
            for block in data["blocks"]:
                if block.get("type") != 0 or _in_table(block):
                    continue
                y0 = block["bbox"][1]
                lines = []
                for line in block["lines"]:
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if text:
                        lines.append((text, max(s["size"] for s in line["spans"] if s["text"].strip())))
                if not lines:
                    continue

                buf_text = ""
                for text, size in lines:
                    if _looks_like_heading(text, size, body_size):
                        if buf_text:
                            items.append((y0, Block(buf_text, "paragraph", pno)))
                            buf_text = ""
                        items.append((y0, Block(_clean_heading(text), "heading", pno,
                                                1 if size >= body_size * 1.35 else 2)))
                    elif is_heading_text(text) and len(text) <= 30:
                        if buf_text:
                            items.append((y0, Block(buf_text, "paragraph", pno)))
                            buf_text = ""
                        items.append((y0, Block(_clean_heading(text), "heading", pno, 1)))
                    else:
                        buf_text = _smart_join(buf_text, text)
                if buf_text:
                    items.append((y0, Block(buf_text, "paragraph", pno)))

            items.sort(key=lambda p: p[0])
            pd.blocks.extend(b for _, b in items)
    finally:
        pdf.close()
    pd.blocks = _merge_blocks(pd.blocks)
    return pd


def _merge_blocks(blocks: list[Block]) -> list[Block]:
    """后处理：合并被 MuPDF 拆开的连续段落块；合并跨页续表（同列数且续表不带表头）。"""
    merged: list[Block] = []
    for b in blocks:
        prev = merged[-1] if merged else None
        if (prev and prev.kind == "table" and b.kind == "table"
                and b.page == (prev.page or 0) + 1
                and prev.rows and b.rows and b.header is None
                and len(prev.rows[0]) == len(b.rows[0])):
            prev.rows.extend(b.rows)
            prev.text = render_table(prev.header, prev.rows)
        elif (prev and prev.kind == "paragraph" and b.kind == "paragraph"
                and prev.page == b.page and prev.text
                and not prev.text.endswith(tuple("。！？；：.!?"))):
            prev.text = _smart_join(prev.text, b.text)
        else:
            merged.append(b)
    return merged


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
    cjk = lambda ch: "\u4e00" <= ch <= "\u9fff" # noqa: E731
    sep = "" if (cjk(a[-1]) or cjk(b[:1]) or b[:1] in "，。、；：！？（）") else " "
    return a + sep + b


# ---------- Word (.docx) ----------


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
            grid = [[c.text.strip().replace("\n", " ") for c in row.cells] for row in table.rows]
            grid = [r for r in grid if any(x.strip() for x in r)]
            if not grid:
                continue
            header = grid[0] if len(grid) > 1 and all(x.strip() for x in grid[0]) else None
            rows = grid[1:] if header else grid
            pd.blocks.append(Block(render_table(header, rows), "table", None,
                                   rows=rows, header=header))
    return pd


# ---------- TXT / Markdown ----------


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
        if first.startswith(" #"):
            level = max(1, len(first) - len(first.lstrip(" #")))
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


# ---------- 统一入口 ----------

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
