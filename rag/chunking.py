"""文档切分。

smart_chunk：结构感知切分——
  1. 依据标题还原「文档 > 章节」层级，切分不跨章节；
  2. 章节内按段落贪心打包到 chunk_size，超长段落按句子边界下切；
  3. 相邻块保留 1 句重叠，避免答案恰好被切分点截断；
  4. 每个 chunk 携带 doc_name / section_path / page 元数据（溯源依据），
     并在向量入库时拼接「上下文头」提升检索质量。

naive_chunk：固定窗口滑动的朴素切分，仅用于评测对比基线。
"""

import hashlib
import re
from dataclasses import dataclass, field

from .config import CHUNK_SIZE, MIN_CHUNK_CHARS, NAIVE_CHUNK_SIZE, NAIVE_STRIDE, OVERLAP_SENTENCES
from .config import ACL_DEFAULT_TAGS
from .parsers import ParsedDoc, render_table

_RE_SENT = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|\n+|$)")

# 修复：原先这两个阈值以魔法数字形式散落在代码里（chunk_size * 0.4、chunk_size + 120），
# 无法统一调整、含义也不直观，现提为具名常量。
OVERLAP_MAX_RATIO = 0.4 # 携带的句子级重叠最多占块大小的比例（防止重叠喧宾夺主）
MERGE_SLACK_CHARS = 120 # 过短块并入前一块时允许的额外长度余量


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    section_path: str # 例："入职与试用期 > 试用期"，空字符串表示无结构
    page: int | None
    text: str
    # 文档级权限标签（ACL）。入库时写入向量库 metadata，检索时按用户组过滤。
    # 默认 ["public"]；敏感文档应在入库时显式指定，如 ["hr"]、["finance","admin"]。
    perm_tags: list[str] = field(default_factory=lambda: list(ACL_DEFAULT_TAGS))
    # 制度类文档的生效/失效元数据：用于过滤已废止文档、以及冲突时优先新版
    effective_from: str = ""
    effective_to: str = ""
    doc_version: str = ""

    @property
    def header(self) -> str:
        loc = f"{self.doc_name} · {self.section_path}" if self.section_path else self.doc_name
        return f"【{loc}】"

    def embed_text(self) -> str:
        """向量入库/检索用的文本：拼接文档名与章节路径作为上下文增强。"""
        return f"{self.header}\n{self.text}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _RE_SENT.findall(text) if s.strip()]


# ---------- 智能切分 ----------


def table_groups(header: list[str] | None, rows: list[list[str]], budget: int) -> list[str]:
    """大表格按行分组渲染，每组独立携带表头并带【表格】标记——
    保证任一组被召回时都有列语义，且生成端能识别这是表格片段。"""
    if not rows:
        return []
    groups: list[str] = []
    cur: list[list[str]] = []
    for r in rows:
        cur.append(r)
        if len(render_table(header, cur)) >= budget:
            groups.append("【表格】" + render_table(header, cur))
            cur = []
    if cur:
        groups.append("【表格】" + render_table(header, cur))
    return groups


def smart_chunk(
    doc: ParsedDoc,
    chunk_size: int = CHUNK_SIZE,
    overlap_sentences: int = OVERLAP_SENTENCES,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    # 1) 标题 → 章节路径，收集各章节的 (text, page) 单元
    sections: list[tuple[list[str], list[tuple[str, int | None]]]] = []
    path: list[str] = []
    for b in doc.blocks:
        if b.kind == "heading":
            level = b.level or 1
            path = path[: level - 1] + [b.text]
            sections.append((path.copy(), []))
        else:
            if not sections:
                sections.append(([], []))
            if b.kind == "table" and b.rows:
                # 表格特殊处理：不跨表格切分；大表按行分组，每组带表头上下文
                for gtext in table_groups(b.header, b.rows, budget=max(120, chunk_size // 2)):
                    sections[-1][1].append((gtext, b.page))
            else:
                for p in re.split(r"\n{2,}", b.text):
                    p = p.strip()
                    if p:
                        sections[-1][1].append((p, b.page))

    # 2) 章节内贪心打包；超长段落按句子边界下切；块间保留句子级重叠
    chunks: list[Chunk] = []

    def new_chunk_id(doc_name: str, idx: int) -> str:
        tag = hashlib.md5(doc_name.encode("utf-8")).hexdigest()[:6]
        return f"{tag}-{idx:04d}"

    idx = 0
    for sec_path, units in sections:
        flat: list[tuple[str, int | None]] = []
        for text, page in units:
            if len(text) > chunk_size:
                flat.extend((s, page) for s in split_sentences(text))
            else:
                flat.append((text, page))

        cur: list[tuple[str, int | None]] = []
        cur_len = 0

        def flush() -> None:
            nonlocal idx, cur, cur_len
            if not cur:
                return
            text = "\n".join(t for t, _ in cur).strip()
            page = next((pg for _, pg in cur if pg is not None), None)
            chunks.append(Chunk(new_chunk_id(doc.doc_name, idx), doc.doc_name,
                                " > ".join(sec_path[-3:]), page, text))
            idx += 1
            cur, cur_len = [], 0

        for text, page in flat:
            if cur and cur_len + len(text) + 1 > chunk_size:
                flush()
                # 重叠：取上一块结尾句子，避免答案在切分点被截断
                # 修复：overlap_sentences=0 时切片 `[-0:]` 等价于 `[0:]`，会把**整块**前文
                # 当作重叠内容重复塞进新块（虽然下面的长度护栏通常能兜住，但属于明确的逻辑错误）。
                prev = chunks[-1].text if chunks else ""
                tail = ("\n".join(split_sentences(prev)[-overlap_sentences:])
                        if prev and overlap_sentences > 0 else "")
                if tail and len(tail) <= chunk_size * OVERLAP_MAX_RATIO:
                    cur, cur_len = [(tail, page)], len(tail)
            cur.append((text, page))
            cur_len += len(text) + 1
        flush()

    # 3) 过短块并入同章节的前一块
    merged: list[Chunk] = []
    for c in chunks:
        if merged and len(c.text) < min_chars and c.section_path == merged[-1].section_path \
                and len(merged[-1].text) + len(c.text) <= chunk_size + MERGE_SLACK_CHARS:
            merged[-1].text = merged[-1].text + "\n" + c.text
        else:
            merged.append(c)
    return merged


# ---------- 朴素切分（评测基线） ----------


def naive_chunk(text: str, doc_name: str, size: int = NAIVE_CHUNK_SIZE, stride: int = NAIVE_STRIDE) -> list[Chunk]:
    """固定窗口滑动切分：无结构感知、无章节信息，仅按字数推进。"""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    tag = hashlib.md5(doc_name.encode("utf-8")).hexdigest()[:6]
    idx = 0
    start = 0
    while start < len(text):
        piece = text[start : start + size]
        chunks.append(Chunk(f"{tag}-{idx:04d}", doc_name, "", None, piece))
        idx += 1
        if start + size >= len(text):
            break
        start += stride
    return chunks


def flatten_doc_text(doc: ParsedDoc) -> str:
    """把解析结果压平成纯文本（供朴素切分等基线使用）。"""
    parts = [b.text for b in doc.blocks if b.text.strip()]
    return "\n".join(parts)
