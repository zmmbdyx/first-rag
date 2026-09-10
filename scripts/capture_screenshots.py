"""用 Playwright 自动操作 Streamlit 问答界面并截图（供 README 展示）。

前置：
    pip install playwright && playwright install chromium
    streamlit run app.py --server.port 8501 --server.headless true

用法：
    python scripts/capture_screenshots.py [--url http://127.0.0.1:8501] [--question "试用期多长时间？"]

产出（保存到项目根目录 screenshots/）：
    rag_main.png     主界面（欢迎页 + 侧边栏配置）
    rag_answering.png 检索/流式生成过程
    rag_result.png   回答结果 + 引用来源展开
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "screenshots"


def wait_idle(page, timeout_ms: int = 180_000) -> None:
    """等待 Streamlit 脚本执行结束（没有 "RUNNING" 状态标记）。"""
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=timeout_ms,
    )


def scroll_last_message_top(page) -> None:
    """把最后一条聊天消息的顶部对齐到视口顶部（Streamlit 的滚动容器在 stMain 上）。"""
    page.evaluate(
        """() => {
            const msgs = document.querySelectorAll('[data-testid="stChatMessage"]');
            const el = msgs[msgs.length - 1];
            if (el) el.scrollIntoView({block: 'start', behavior: 'instant'});
            const main = document.querySelector('[data-testid="stMain"]');
            if (main) main.scrollTop = Math.max(0, main.scrollTop - 20);
        }"""
    )
    page.wait_for_timeout(1200)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--question", default="试用期多长时间？")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 950},
                                device_scale_factor=1.5)
        print(f"→ 打开 {args.url}")
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        ***REMOVED*** 首页需要加载嵌入模型 + 构建 BM25（首次约 10-30s）
        page.wait_for_selector("text=智能知识库问答", timeout=args.timeout * 1000)
        wait_idle(page)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT_DIR / "rag_main.png"))
        print("  ✅ rag_main.png")

        ***REMOVED*** ---- 提问：触发检索 + 流式生成 ----
        box = page.locator('[data-testid="stChatInput"] textarea').first
        box.click()
        box.fill(args.question)
        box.press("Enter")

        ***REMOVED*** 运行过程：检索 spinner 与流式回答都在这一刻出现
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT_DIR / "rag_answering.png"))
        print("  ✅ rag_answering.png")

        ***REMOVED*** 等回答结束（无 RUNNING 标记且出现耗时说明）
        try:
            page.wait_for_selector("text=总耗时", timeout=args.timeout * 1000)
        except Exception:  ***REMOVED*** noqa: BLE001 — 模型不可用时也要留下结果截图
            print("  ⚠️  未捕获到「总耗时」，可能模型调用失败，仍继续截图")
        time.sleep(1.5)

        ***REMOVED*** 把最后一条助手回复滚到视口顶部（block:'start'），保证"答案 + 耗时说明"完整入镜
        scroll_last_message_top(page)
        page.screenshot(path=str(OUT_DIR / "rag_result.png"))
        print("  ✅ rag_result.png")

        ***REMOVED*** ---- 展开引用来源，单独出一张「答案溯源」截图 ----
        expander = page.locator('details:has-text("查看引用来源")').first
        if expander.count():
            expander.click()
            ***REMOVED*** 展开后 Streamlit 需要一次增量渲染把来源列表推下来，等列表出现再截图
            try:
                page.wait_for_function(
                    "() => { const d = [...document.querySelectorAll('details')]"
                    ".find(x => x.textContent.includes('查看引用来源'));"
                    "return !!d && d.querySelectorAll('div[data-testid=\"stMarkdown\"]').length > 0; }",
                    timeout=30_000)
            except Exception:  ***REMOVED*** noqa: BLE001
                pass
            page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT_DIR / "rag_sources.png"))
            print("  ✅ rag_sources.png")

        browser.close()
    print(f"\n🎉 截图已保存到 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
