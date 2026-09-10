"""切分与解析的单元测试（不依赖嵌入模型与大模型，可离线跑）。"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chunking import naive_chunk, smart_chunk, split_sentences
from rag.parsers import Block, ParsedDoc, is_heading_text, parse_txt


def _fake_doc() -> ParsedDoc:
    long_para = "这是第一个句子。这是第二个句子。" * 40 # ~640 字，超过 chunk_size
    return ParsedDoc("测试文档.pdf", "测试文档.pdf", blocks=[
        Block("第一章 制度", "heading", 1, 1),
        Block(long_para, "paragraph", 1),
        Block("第二章 考勤", "heading", 2, 1),
        Block("短内容。", "paragraph", 2),
    ])


def test_split_sentences():
    sents = split_sentences("你好。世界！end without dot")
    assert sents == ["你好。", "世界！", "end without dot"]


def test_is_heading_text():
    assert is_heading_text("第四章 请假制度")
    assert is_heading_text("6.2 差旅标准")
    assert is_heading_text("Q1：忘记OA系统密码怎么办")
    assert not is_heading_text("这是一句普通的话，很长很长很长很长很长很长很长很长。")
    assert not is_heading_text("1350W额定功率，噪音小于58分贝。")


def test_smart_chunk_respects_sections():
    chunks = smart_chunk(_fake_doc(), chunk_size=200, overlap_sentences=1)
    assert len(chunks) >= 4
    # 不允许跨章节：第一章的块不能混入第二章标题之后的文本
    sec1 = [c for c in chunks if c.section_path == "第一章 制度"]
    sec2 = [c for c in chunks if c.section_path == "第二章 考勤"]
    assert sec1 and sec2
    assert all("短内容" not in c.text for c in sec1)
    # 每块都带溯源元数据
    for c in chunks:
        assert c.doc_name == "测试文档.pdf"
        assert c.section_path and c.chunk_id


def test_smart_chunk_overlap():
    chunks = smart_chunk(_fake_doc(), chunk_size=200, overlap_sentences=1)
    long_chunks = [c for c in chunks if c.section_path == "第一章 制度"]
    if len(long_chunks) >= 2:
        tail = long_chunks[0].text.strip().splitlines()[-1]
        assert any(tail[:6] in c.text for c in long_chunks[1:]), "相邻块应有句子级重叠"


def test_naive_chunk_covers_all_text():
    text = "字" * 1000
    chunks = naive_chunk(text, "t.txt", size=200, stride=150)
    assert chunks[0].text == text[:200]
    assert chunks[-1].text.endswith(text[-50:]) # 尾部完整覆盖
    assert len(chunks) == math.ceil((1000 - 200) / 150) + 1 # 首块 + 滑窗次数


def test_parse_txt_faq(tmp_path: Path):
    p = tmp_path / "faq.txt"
    p.write_text("服务FAQ\n\nQ1：怎么重置密码？\n答：到IT服务台重置。\n\nQ2：WiFi密码在哪看？\n答：OA首页公告。\n", encoding="utf-8")
    doc = parse_txt(p)
    kinds = [(b.kind, b.text) for b in doc.blocks]
    assert ("heading", "Q1：怎么重置密码？") in kinds
    assert ("paragraph", "答：到IT服务台重置。") in kinds


def test_chunk_ids_stable():
    ids1 = [c.chunk_id for c in smart_chunk(_fake_doc())]
    ids2 = [c.chunk_id for c in smart_chunk(_fake_doc())]
    assert ids1 == ids2 # 幂等入库依赖稳定 ID


def test_smart_chunk_zero_overlap_no_duplication():
    """回归测试：overlap_sentences=0 曾因切片 `[-0:]` 等价于 `[0:]`
    而把整块前文当作重叠内容重复塞进新块，导致 chunk 数锐减/内容成倍重复。"""
    no_overlap = smart_chunk(_fake_doc(), chunk_size=200, overlap_sentences=0)
    with_overlap = smart_chunk(_fake_doc(), chunk_size=200, overlap_sentences=1)
    # 关掉重叠后总块数不应少于开启重叠时，且正文总长度不应异常膨胀
    assert len(no_overlap) >= len(with_overlap)
    total = sum(len(c.text) for c in no_overlap)
    max_expected = 640 * 2 # 原始正文约 640 字，留出章节标题与连接的余量
    assert total < max_expected, f"关闭重叠后正文被重复填充：{total} 字"
