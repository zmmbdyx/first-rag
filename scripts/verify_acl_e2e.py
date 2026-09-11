"""端到端验证文档级 ACL：**不同用户组通过 HTTP 检索到的文档确实不同**。

这是对 P0「越权检索」的回归守卫。离线单测只能验证 ``is_visible()`` 的判定，
本脚本走真实 API：上传一份带密级的文档 → 用不同 ``X-User-Groups`` 提问 →
断言越权组的引用来源里**完全不出现**该文档。

    python scripts/verify_acl_e2e.py [base_url]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
ROOT = Path(__file__).resolve().parent.parent

FAILED: list[str] = []
SECRET_DOC = "acl_secret_salary_policy.txt"
# 造一份内容独特的文档，便于断言"检索得到 / 检索不到"
SECRET_TEXT = """星辰科技薪酬保密细则

一、薪酬带宽
P7 职级年度现金总额区间为 120 万至 180 万元，含基本工资、绩效奖金与股票。
P8 职级年度现金总额区间为 200 万至 300 万元。

二、保密要求
薪酬信息属最高密级，仅 HRBP 与直属总监可见，禁止任何形式的对外披露。
违反者按严重违纪处理。
"""


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}]   {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def post(path: str, payload: dict, groups: str = "") -> dict:
    r = urllib.request.Request(f"{BASE}{path}",
                               data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                               method="POST")
    r.add_header("Content-Type", "application/json; charset=utf-8")
    if groups:
        r.add_header("X-User-Groups", groups)
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload(path: Path, perm_tags: str, groups: str = "admin") -> dict:
    boundary = "----acle2e"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        b"Content-Type: text/plain\r\n\r\n",
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    url = f"{BASE}/api/upload"
    if perm_tags:
        url += f"?perm_tags={urllib.parse.quote(perm_tags)}"
    r = urllib.request.Request(url, data=body, method="POST")
    r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    r.add_header("X-User-Groups", groups)
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ask(question: str, groups: str) -> tuple[str, list[dict]]:
    """提问并返回 (回答, 引用来源)。"""
    conv = post("/api/conversations", {}, groups)
    cid = conv["conversation_id"]
    payload = {"message": question, "conversation_id": cid, "use_cache": False}
    r = urllib.request.Request(f"{BASE}/api/chat",
                               data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                               method="POST")
    r.add_header("Content-Type", "application/json; charset=utf-8")
    if groups:
        r.add_header("X-User-Groups", groups)
    event, content, sources = "message", [], []
    with urllib.request.urlopen(r, timeout=240) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "content":
                    content.append(data.get("delta", ""))
                elif event == "sources":
                    sources = data
            elif not line:
                event = "message"
    # 清理会话
    try:
        req = urllib.request.Request(f"{BASE}/api/conversations/{cid}", method="DELETE")
        req.add_header("X-User-Groups", groups)
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:  # noqa: BLE001
        pass
    return "".join(content), sources


def doc_names(sources: list[dict]) -> set[str]:
    return {s.get("doc_name", "") for s in sources}


def mentions_secret(sources: list[dict]) -> bool:
    """引用来源里是否提到这份密级文档。

    注意用**子串**匹配：上传落盘时会加 ``时间戳_随机串_`` 前缀防同名覆盖，
    而检索侧展示时会剥掉该前缀，两边的名字并不完全相等。用子串匹配可以
    同时覆盖"展示名"与"库内存储名"两种情况，避免因为名字格式差异而假绿。
    """
    stem = SECRET_DOC.rsplit(".", 1)[0]
    return any(stem in n for n in doc_names(sources))


def list_docs(groups: str = "admin") -> list[dict]:
    r = urllib.request.Request(f"{BASE}/api/documents")
    r.add_header("X-User-Groups", groups)
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))["documents"]


def delete_doc(name: str, groups: str = "admin") -> bool:
    import urllib.parse as _up

    req = urllib.request.Request(
        f"{BASE}/api/documents/{_up.quote(name)}", method="DELETE")
    req.add_header("X-User-Groups", groups)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8")).get("ok") is True
    except urllib.error.HTTPError:
        return False


def purge_previous_copies() -> int:
    """清掉历史上遗留的同名文档副本。

    上传落盘会带 ``时间戳_随机串_`` 前缀（防同名覆盖），因此**同名文档可以
    在库里存在多份**。如果上一次运行中途失败没清理，残留副本会让
    「删除后不再被检索到」这条断言假失败 —— 删掉的是本次上传的那份，
    检索命中的是上一次的残留。这里先做一次彻底清理，保证用例可重复运行。
    """
    removed = 0
    for row in list_docs():
        if SECRET_DOC in row["doc_name"]:
            if delete_doc(row["doc_name"]):
                removed += 1
    return removed


def main() -> int:
    import urllib.parse  # noqa: F401  放在此处避免顶部未使用告警

    print("=== 0) 准备 ===")
    purged = purge_previous_copies()
    if purged:
        print(f"  （清理了 {purged} 份历史残留副本，保证本次断言有效）")

    tmp = ROOT / "data" / "samples" / SECRET_DOC
    tmp.write_text(SECRET_TEXT, encoding="utf-8")
    try:
        up = upload(tmp, perm_tags="hr")
        check("上传成功", up.get("status") == "done", json.dumps(up, ensure_ascii=False))
    finally:
        tmp.unlink(missing_ok=True)

    question = "P8职级的年度现金总额区间是多少？"

    print("\n=== 1) hr 组应当检索到该文档 ===")
    ans_hr, src_hr = ask(question, groups="hr")
    names_hr = doc_names(src_hr)
    check("hr 组能看到密级文档", mentions_secret(src_hr), f"来源={sorted(names_hr)}")
    check("hr 组的回答非空", len(ans_hr) > 0, f"{len(ans_hr)} 字")

    print("\n=== 2) it 组**不得**检索到该文档（核心断言）===")
    ans_it, src_it = ask(question, groups="it")
    names_it = doc_names(src_it)
    check("it 组的来源中不含密级文档", not mentions_secret(src_it),
          f"来源={sorted(names_it)}")
    check("it 组的回答未泄露薪酬数字",
          "120" not in ans_it and "200" not in ans_it and "180" not in ans_it,
          ans_it[:120])

    print("\n=== 3) admin 组可见全部 ===")
    _, src_ad = ask(question, groups="admin")
    check("admin 组能看到密级文档", mentions_secret(src_ad),
          f"来源={sorted(doc_names(src_ad))}")

    print("\n=== 4) 文档清单与密级 ===")
    stem = SECRET_DOC.rsplit(".", 1)[0]
    catalog = list_docs()
    rows = [d for d in catalog if stem in d["doc_name"]]
    check("清单里恰好一份该文档", len(rows) == 1, f"命中 {len(rows)} 份"),
    check("清单里该文档密级为 hr", bool(rows) and rows[0]["perm_tags"] == ["hr"],
          json.dumps(rows[0], ensure_ascii=False) if rows else "未找到")
    stored_name = rows[0]["doc_name"] if rows else SECRET_DOC

    print("\n=== 5) 清理：删除该文档（合规删除路径）===")
    # 用**库内存储名**删除；接口也支持用展示名（会做一次容错解析）
    check("删除成功", delete_doc(stored_name))

    # 删除后再问一次，确认向量确已被移除
    _, src_after = ask(question, groups="hr")
    check("删除后不再被检索到", not mentions_secret(src_after),
          f"来源={sorted(doc_names(src_after))}")

    print()
    if FAILED:
        print(f"[失败] {len(FAILED)} 项未通过：")
        for f in FAILED:
            print("   -", f)
        return 1
    print("[通过] 文档级 ACL 与删除路径端到端验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
