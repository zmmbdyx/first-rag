# -*- coding: utf-8 -*-
"""端到端验证：/ingest 上传入库 → 缓存按 KB 版本失效 → 新文档可被检索到。

用法（需先启动 API）：
    uvicorn api_server:app --port 8011
    python scripts/verify_api_e2e.py --base http://127.0.0.1:8011
"""
import argparse
import json
import sys
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def post_json(url, payload, timeout=300):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post_file(url, filename, content: bytes, timeout=600):
    b = uuid.uuid4().hex
    head = (f'--{b}\r\nContent-Disposition: form-data; name="files"; '
            f'filename="{filename}"\r\nContent-Type: text/plain\r\n\r\n').encode("utf-8")
    body = b"".join([head, content, b"\r\n", f"--{b}--\r\n".encode()])
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    from rag import cache as qa_cache

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        if not cond:
            ok = False
        print(("  ✅ " if cond else "  ❌ ") + name + (f"  ({detail})" if detail else ""))

    print("== 1. 健康检查 ==")
    with urllib.request.urlopen(f"{base}/health", timeout=60) as r:
        h = json.loads(r.read().decode("utf-8"))
    check("status ok", h["status"] == "ok", h["status"])
    check("已入库文档块", h["chunks"] and h["chunks"] > 0, str(h["chunks"]))
    check("BM25 就绪", h["bm25_ready"])
    print(f"     缓存: enabled={h['cache']['enabled']} url={h['cache']['url']}")

    print("== 2. 入库前：查一个知识库里没有的探针问题 ==")
    probe_q = "探针编号 PROBE-9X7 的有效期是多少个月？"
    r1 = post_json(f"{base}/ask", {"question": probe_q, "mode": "hybrid", "k": 5})
    check("入库前检索不到（拒答/空答案）",
          (not r1["citations"]) or ("未找到" in r1["answer"]),
          f"citations={r1['citations']}")

    ver_before = qa_cache.kb_version() if qa_cache.available() else 0
    print(f"     入库前 kb_version={ver_before}")

    print("== 3. 上传并入库探针文档 ==")
    name = f"ingest_probe_{uuid.uuid4().hex[:6]}.txt"
    content = (
        "第一章 总则\n本文档用于验证 /ingest 端点与缓存失效。\n\n"
        "第二章 关键参数\n探针编号 PROBE-9X7，有效期为 42 个月。\n"
    ).encode("utf-8")
    d = post_file(f"{base}/ingest?incremental=true&workers=4", name, content)
    check("入库成功且产生新块", d["new_chunks"] > 0, f"new_chunks={d['new_chunks']}")
    check("返回的 cache_version 已自增", d["cache_version"] > ver_before,
          f"{ver_before} -> {d['cache_version']}")

    if qa_cache.available():
        check("KB 版本号已自增", qa_cache.kb_version() > ver_before,
              f"{ver_before} -> {qa_cache.kb_version()}")
        left = list(qa_cache._client.scan_iter(match=f"{qa_cache.CACHE_PREFIX}:v*:*", count=500))
        # 旧版本的键应被清理（新版本刚自增，尚无写入）
        old_left = [k for k in left if f":v{ver_before}:" in k]
        check("旧版本缓存键已被清理", not old_left, f"剩余 {len(old_left)} 条旧键")

    print("== 4. 入库后：同一问题应能检索到新文档 ==")
    r2 = post_json(f"{base}/ask", {"question": probe_q, "mode": "hybrid", "k": 5})
    check("检索到新上传文档",
          any("ingest_probe" in (s.get("doc_name") or "") for s in r2["sources"]),
          str([s.get("doc_name") for s in r2["sources"]][:3]))
    check("答案包含探针要点", "42" in r2["answer"], r2["answer"][:60])

    print("\n" + "=" * 46)
    print("端到端验证通过 ✅" if ok else "存在失败项 ❌")
    print(f"（探针文档已入库到 data/uploads/{name}，如需清理可手动删除）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
