"""界面截图：空状态 / 问答态 / 深色模式 / 引用来源。

    python scripts/capture_frontend.py [frontend_url]

产物写入 screenshots/frontend/，用于 README 与人工验收。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots" / "frontend"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5173"
QUESTION = "员工手册里试用期是多久？"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        page.on("console", lambda m: errors.append(f"[console.{m.type}] {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))

        print(f"打开 {URL} …")
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(1500)

        # ---- 1) 空状态（浅色）----
        page.screenshot(path=str(OUT / "01-empty-light.png"))
        print("  ✓ 01-empty-light.png")

        # 主题与欢迎语是否渲染
        body = page.inner_text("body")
        for needle in ("知识库智能问答", "今天有什么可以帮到你"):
            if needle not in body:
                errors.append(f"空状态缺少文案：{needle}")

        # ---- 2) 点击推荐问题，观察流式生成 ----
        card = page.get_by_text("员工手册里试用期是多久？", exact=True).first
        if card.count() > 0:
            card.click()
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / "02-streaming.png"))
            print("  ✓ 02-streaming.png（流式生成中）")

            # 等回答完成（操作栏出现 = 生成结束）
            try:
                page.wait_for_selector("button[aria-label='重新生成']", timeout=120_000)
            except Exception as e:  # noqa: BLE001
                errors.append(f"等待回答完成超时：{e}")
            page.wait_for_timeout(1200)

            page.screenshot(path=str(OUT / "03-answer-light.png"))
            print("  ✓ 03-answer-light.png")

            # 滚动到引用来源
            page.mouse.wheel(0, 900)
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / "04-sources.png"))
            print("  ✓ 04-sources.png")

            # 展开一张来源卡片
            src = page.locator("button[aria-expanded]").filter(has_text="相似度").first
            if src.count() > 0:
                src.click()
                page.wait_for_timeout(500)
                page.screenshot(path=str(OUT / "05-source-expanded.png"))
                print("  ✓ 05-source-expanded.png")
        else:
            errors.append("未找到推荐问题卡片")

        # ---- 3) 深色模式 ----
        page.get_by_role("button", name="深色模式").first.click()
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "06-answer-dark.png"))
        print("  ✓ 06-answer-dark.png")

        # 深色下回到空状态看看欢迎页
        page.get_by_text("开启新对话").first.click()
        page.wait_for_timeout(900)
        page.screenshot(path=str(OUT / "07-empty-dark.png"))
        print("  ✓ 07-empty-dark.png")

        # ---- 4) 移动端布局 ----
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mobile.goto(URL, wait_until="networkidle", timeout=60_000)
        mobile.wait_for_timeout(1200)
        mobile.screenshot(path=str(OUT / "08-mobile-empty.png"))
        print("  ✓ 08-mobile-empty.png")

        mobile.get_by_role("button", name="切换侧边栏").click()
        mobile.wait_for_timeout(700)
        mobile.screenshot(path=str(OUT / "09-mobile-sidebar.png"))
        print("  ✓ 09-mobile-sidebar.png")

        browser.close()

    print()
    if errors:
        print(f"[警告] 发现 {len(errors)} 个问题：")
        for e in errors[:20]:
            print("   -", e)
        return 1
    print("[通过] 截图完成，控制台无错误")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
