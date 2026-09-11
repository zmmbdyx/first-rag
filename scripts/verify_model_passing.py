"""验证「模型下拉框是否真的生效」。

后端 .env 的 LLM_MODEL 是默认模型；前端下拉框有 4 个选项。
本脚本在浏览器里把模型切成**与默认不同的另一项**，发一次提问，
然后检查 /api/chat 请求体里的 model 字段，以及运行指标落库的 model 值。

此前前端根本没把 model 发出去，下拉框形同虚设——本脚本就是这条回归的守卫。
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.config import LLM_MODEL  # noqa: E402
from rag.metrics import DB_PATH  # noqa: E402

URL = "http://127.0.0.1:5173"
FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}]   {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def metric_rows() -> list[tuple]:
    if not DB_PATH.exists():
        return []
    c = sqlite3.connect(DB_PATH)
    try:
        return c.execute("select ts, model, mode from chat_metrics order by rowid").fetchall()
    finally:
        c.close()


def main() -> int:
    print(f"后端 .env 默认模型: {LLM_MODEL}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        sent: list[dict] = []

        def on_request(req):
            if req.url.endswith("/api/chat") and req.method == "POST":
                try:
                    sent.append(json.loads(req.post_data or "{}"))
                except Exception:  # noqa: BLE001
                    pass

        page.on("request", on_request)
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(1500)

        # 打开模型下拉，选一个与默认不同的选项
        trigger = page.locator("button[aria-haspopup='listbox']").first
        if trigger.count() == 0:
            check("找到模型下拉框", False)
            browser.close()
            return 1
        check("找到模型下拉框", True, (trigger.inner_text() or "").strip())

        trigger.click()
        page.wait_for_timeout(500)
        options = page.locator("div[role='listbox'] button[role='option']")
        n = options.count()
        check("下拉框有多个模型可选", n > 1, f"{n} 个")
        if n == 0:
            browser.close()
            return 1

        labels = [options.nth(i).inner_text().strip() for i in range(n)]
        print(f"  可选模型: {labels}")
        # 明确挑一个与后端 .env 默认值**不同**的模型（不靠 ✓ 字形判断，
        # 那在不同字体/渲染下不可靠），这样才真正验证"切换生效"。
        want = next((i for i, lb in enumerate(labels) if LLM_MODEL not in lb), -1)
        check("存在与默认不同的可选项", want >= 0,
              f"将选下标 {want} = {labels[want] if want >= 0 else None!r}")
        if want < 0:
            browser.close()
            return 1
        target_idx, target_label = want, labels[want]
        options.nth(target_idx).click()
        page.wait_for_timeout(600)
        shown = (trigger.inner_text() or "").strip()
        print(f"  已切换到: {target_label}  （下拉框显示: {shown}）")
        check("下拉框显示已切换", LLM_MODEL not in shown, f"显示={shown!r}")

        before = metric_rows()
        # 提问
        page.get_by_text("员工手册里试用期是多久？", exact=True).first.click()
        try:
            page.wait_for_selector("button[aria-label='重新生成']", timeout=180_000)
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(1500)

        check("捕获到 /api/chat 请求", len(sent) > 0, f"{len(sent)} 次")
        if sent:
            payload = sent[-1]
            check("请求体带 model 字段", "model" in payload,
                  f"model={payload.get('model')!r}")
            check("model 非空（下拉框真的生效）", bool(payload.get("model")),
                  repr(payload.get("model")))
            # 关键：发出去的必须是**切换后**的模型，而不是 .env 默认值
            sent_model = str(payload.get("model") or "")
            check("发出的模型 = 界面所选（不是默认回落）",
                  sent_model and sent_model != LLM_MODEL,
                  f"发出={sent_model!r}  默认={LLM_MODEL!r}")
            check("请求体带 mode", "mode" in payload, repr(payload.get("mode")))
            check("请求体带 top_k", "top_k" in payload, repr(payload.get("top_k")))

        after = metric_rows()
        added = after[len(before):]
        print(f"\n  新增指标行: {added}")
        if added:
            check("指标里 model 非空", bool(added[-1][1]), repr(added[-1][1]))
        else:
            check("指标新增 1 行", False, "没有新增")

        browser.close()

    print()
    if FAILED:
        print(f"[失败] {len(FAILED)} 项未通过：")
        for f in FAILED:
            print("   -", f)
        return 1
    print("[通过] 模型选择与检索参数确实随请求发出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
