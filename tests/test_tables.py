"""表格结构处理测试：渲染、按行分组携带表头、跨页合并、docx 结构化解析。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chunking import smart_chunk, table_groups  ***REMOVED*** noqa: E402
from rag.parsers import Block, ParsedDoc, _merge_blocks, parse_file, render_table  ***REMOVED*** noqa: E402


def test_render_table_kv_mode():
    text = render_table(["项目", "参数"], [["额定功率", "1350W"], ["净重", "7.8kg"]])
    assert "项目 | 参数" in text
    assert "额定功率=1350W" in text and "净重=7.8kg" in text


def test_render_table_general_header_carry():
    text = render_table(["主机名", "机房"], [["db-01", "上海金桥"]])
    assert "主机名=db-01 | 机房=上海金桥" in text


def test_table_groups_carry_header():
    groups = table_groups(["型号", "价格"], [[f"产品{i}", f"{i}元"] for i in range(12)], budget=60)
    assert len(groups) >= 2
    for g in groups:
        assert "型号 | 价格" in g.splitlines()[0]  ***REMOVED*** 每组都带表头


def test_merge_cross_page_table():
    t1 = Block(render_table(["里程碑", "交付物"], [["M1", "文档"]]), "table", 1,
               rows=[["M1", "文档"]], header=["里程碑", "交付物"])
    t2 = Block(render_table(None, [["M9", "报告"]]), "table", 2, rows=[["M9", "报告"]])
    t3 = Block(render_table(["x"], [["y"]]), "table", 4, rows=[["y"]], header=["x"])
    merged = _merge_blocks([t1, t2, t3])
    assert len(merged) == 2
    assert merged[0].rows == [["M1", "文档"], ["M9", "报告"]]
    assert merged[0].page == 1  ***REMOVED*** 合并到首页的表


def test_merge_keeps_separate_tables_without_continuation():
    t1 = Block("a", "table", 1, rows=[["1"]], header=["h"])
    t2 = Block("b", "table", 1, rows=[["2"]], header=["h"])  ***REMOVED*** 同页两表不合并
    assert len(_merge_blocks([t1, t2])) == 2


def test_smart_chunk_table_not_split_mid_group():
    rows = [[f"设备{i}", f"型号{i}号", f"{i}万元"] for i in range(30)]
    doc = ParsedDoc("清单.docx", "清单.docx", blocks=[
        Block("服务器清单", "heading", None, 1),
        Block(render_table(["名称", "型号", "价格"], rows), "table", None,
              rows=rows, header=["名称", "型号", "价格"]),
    ])
    chunks = smart_chunk(doc, chunk_size=300)
    assert len(chunks) >= 2
    for c in chunks:
        assert "名称 | 型号 | 价格" in c.text  ***REMOVED*** 每块都带表头上下文


def test_docx_corpus_table_parsed():
    p = ROOT / "data" / "corpus" / "云滴咖啡机C3产品说明书.docx"
    if not p.exists():
        return
    doc = parse_file(p)
    tbl = [b for b in doc.blocks if b.kind == "table"][0]
    assert tbl.header == ["项目", "参数"]
    assert any(r[0] == "额定功率" for r in tbl.rows)
    assert "额定功率=1350W" in tbl.text
