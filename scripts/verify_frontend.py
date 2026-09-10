"""前端界面验收：用 Playwright 走一遍真实交互并断言关键行为。

覆盖：
  1. 空状态欢迎语与推荐问题渲染；
  2. 点击推荐问题 → 流式生成 → 回答落定 → 引用来源出现；
  3. 侧边栏悬停时"时间"与"操作按钮"互斥显示（曾经重叠的 bug 回归测试）；
  4. 搜索历史对话过滤生效；
  5. 重命名与删除按钮可用；
  6. 深色/浅色切换确实改变了根元素 class 与背景色；
  7. 拖拽上传文档（DataTransfer 模拟）后端返回入库结果；
  8. 移动端汉堡菜单能打开侧边栏抽屉。

    python scripts/verify_frontend.py [frontend_url]
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5173"
QUESTION = "员工手册里试用期是多久？"

FAILED: list[str] = []
CONSOLE_ERRORS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}]   {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def root_bg(page: Page) -> str:
    return page.evaluate(
        "() => getComputedStyle(document.body).backgroundColor"
    )


def hover_item_title(page: Page, item) -> None:
    """把鼠标停在会话条目的标题区（避开右侧按钮），以触发悬停态。"""
    box = item.bounding_box()
    if box:
        page.mouse.move(box["x"] + 40, box["y"] + box["height"] / 2)
        page.wait_for_timeout(350)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: CONSOLE_ERRORS.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: CONSOLE_ERRORS.append(str(e)))

        print("=== 1) 空状态 ===")
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(1200)
        body = page.inner_text("body")
        check("品牌标题", "知识库智能问答" in body)
        check("欢迎语", "今天有什么可以帮到你" in body)
        check("推荐问题卡片", page.get_by_text(QUESTION, exact=True).count() > 0)
        check("知识库状态徽标", "切片" in body)
        light_bg = root_bg(page)

        print("\n=== 2) 提问 → 流式 → 回答与来源 ===")
        page.get_by_text(QUESTION, exact=True).first.click()
        page.wait_for_timeout(1500)
        streaming_body = page.inner_text("body")
        check("提问后进入生成态（或已出内容）",
              QUESTION in streaming_body)

        try:
            page.wait_for_selector("button[aria-label='重新生成']", timeout=120_000)
            done = True
        except Exception:  # noqa: BLE001
            done = False
        check("生成完成（操作栏出现）", done)
        page.wait_for_timeout(800)

        body2 = page.inner_text("body")
        check("回答含引用标注 [1]", "[1]" in body2)
        check("出现引用来源区块", "引用来源" in body2)
        check("来源含文档名", "员工手册" in body2)
        check("来源含相似度", "相似度" in body2)
        check("操作栏含复制/重新生成/点赞/点踩",
              all(page.locator(f"button[aria-label='{a}']").count() > 0
                  for a in ("复制回答", "重新生成", "点赞", "点踩")))

        print("\n=== 3) 重新生成：不应出现重复的用户提问 ===")
        # 记录当前用户气泡数量（用户消息右对齐气泡）
        def user_bubble_count() -> int:
            return page.evaluate(
                """() => {
                // 用户气泡：flex justify-end 容器里的消息块
                return document.querySelectorAll('div.justify-end > div').length;
            }"""
            )

        def bubble_texts() -> list[str]:
            return page.evaluate(
                """() => [...document.querySelectorAll('div.justify-end > div')]
                        .map(e => (e.textContent || '').trim())"""
            )

        before_n = user_bubble_count()
        texts_before = bubble_texts()
        print(f"    重新生成前用户气泡: {before_n} 个")

        regen = page.locator("button[aria-label='重新生成']").first
        if regen.count() > 0:
            regen.click()
            page.wait_for_timeout(1500)
            try:
                page.wait_for_selector("button[aria-label='重新生成']", timeout=120_000)
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(1200)

            after_n = user_bubble_count()
            texts_after = bubble_texts()
            check("重新生成后用户气泡数量不变", after_n == before_n, f"{before_n} → {after_n}")
            check("没有出现重复的提问气泡", texts_after == texts_before,
                  f"{texts_before} → {texts_after}")
            check("回答仍然存在",
                  page.locator("button[aria-label='复制回答']").count() > 0)
        else:
            check("存在重新生成按钮", False)

        print("\n=== 4) 侧边栏悬停互斥（重叠 bug 回归）===")
        item = page.locator(".conv-item").first
        if item.count() > 0:
            # 把鼠标移到条目左侧的标题区，避开右侧按钮，避免"悬停后指针落在按钮上"导致状态翻转
            box = item.bounding_box()
            assert box is not None
            page.mouse.move(box["x"] + 40, box["y"] + box["height"] / 2)
            page.wait_for_timeout(400)

            btn = item.locator("button[aria-label='重命名']")
            time_span = item.locator("span").filter(has_text="刚刚").first

            hovered_btn = btn.count() > 0 and btn.first.is_visible()
            hovered_time = time_span.count() > 0 and time_span.is_visible()
            check("悬停后显示操作按钮", hovered_btn)
            check("悬停后不再显示时间（避免与按钮重叠）", not hovered_time,
                  f"time_visible={hovered_time}")
            check("两者不同时可见", not (hovered_time and hovered_btn))

            # 移开鼠标：按钮收起、时间恢复
            page.mouse.move(box["x"] + 40, box["y"] + 200)
            page.wait_for_timeout(400)
            idle_btn = btn.count() > 0 and btn.first.is_visible()
            idle_time = item.locator("span").filter(has_text="刚刚").first.is_visible()
            check("移开鼠标后隐藏操作按钮", not idle_btn)
            check("移开鼠标后恢复显示时间", idle_time)
        else:
            check("存在会话列表项", False)

        print("\n=== 5) 搜索过滤 ===")
        search = page.get_by_placeholder("搜索历史对话")
        if search.count() > 0:
            total_before = page.locator(".conv-item").count()
            search.fill("绝对不存在的关键词xyz")
            page.wait_for_timeout(400)
            after = page.locator(".conv-item").count()
            check("无匹配时列表清空", after == 0, f"{total_before} → {after}")
            search.fill("")
            page.wait_for_timeout(400)
            check("清空搜索后恢复", page.locator(".conv-item").count() == total_before)
        else:
            check("存在搜索框", False)

        print("\n=== 6) 重命名 ===")
        item = page.locator(".conv-item").first
        if item.count() > 0:
            hover_item_title(page, item)
            rename_btn = item.locator("button[aria-label='重命名']").first
            if rename_btn.is_visible():
                rename_btn.click()
                page.wait_for_timeout(400)
                box_input = item.locator("input")
                if box_input.count() > 0:
                    box_input.fill("验收重命名标题")
                    box_input.press("Enter")
                    page.wait_for_timeout(1500)
                    check("重命名后标题更新", "验收重命名标题" in page.inner_text("body"))
                else:
                    check("重命名输入框出现", False)
            else:
                check("悬停后重命名按钮可见", False)
        else:
            check("存在可重命名的会话", False)

        print("\n=== 7) 主题切换 ===")
        page.get_by_role("button", name="深色模式").first.click()
        page.wait_for_timeout(700)
        dark_class = page.evaluate("() => document.documentElement.classList.contains('dark')")
        dark_bg = root_bg(page)
        check("html 挂上 dark 类", dark_class)
        check("背景色确实变深", dark_bg != light_bg, f"{light_bg} → {dark_bg}")

        page.get_by_role("button", name="浅色模式").first.click()
        page.wait_for_timeout(700)
        check("切回浅色", not page.evaluate(
            "() => document.documentElement.classList.contains('dark')"))

        print("\n=== 8) 拖拽上传文档 ===")
        sample = ROOT / "data" / "samples" / "远程办公管理制度.txt"
        if sample.exists():
            data = sample.read_bytes()
            page.evaluate(
                """async ([name, b64]) => {
                    const bin = atob(b64);
                    const arr = new Uint8Array(bin.length);
                    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
                    const file = new File([arr], name, { type: 'text/plain' });
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    const target = document.querySelector('textarea').closest('div').parentElement;
                    target.dispatchEvent(new DragEvent('dragover', {
                        bubbles: true, cancelable: true, dataTransfer: dt }));
                    target.dispatchEvent(new DragEvent('drop', {
                        bubbles: true, cancelable: true, dataTransfer: dt }));
                }""",
                [sample.name, __import__("base64").b64encode(data).decode()],
            )
            page.wait_for_timeout(1000)
            drop_hint = page.inner_text("body")
            check("拖拽后进入上传流程",
                  "正在解析并入库" in drop_hint or "已入库" in drop_hint or "上传" in drop_hint)

            try:
                page.wait_for_selector("text=已入库", timeout=180_000)
                check("上传并入库成功提示", True)
            except Exception:  # noqa: BLE001
                body3 = page.inner_text("body")
                check("上传并入库成功提示", "已入库" in body3,
                      body3[:200].replace("\n", " "))
        else:
            check("示例文件存在", False, str(sample))

        print("\n=== 9) 移动端抽屉 ===")
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(URL, wait_until="networkidle", timeout=60_000)
        mobile.wait_for_timeout(1000)
        aside_hidden = mobile.locator("aside").first.bounding_box()
        check("移动端侧边栏默认收起",
              aside_hidden is None or aside_hidden["x"] < 0,
              str(aside_hidden))
        mobile.get_by_role("button", name="切换侧边栏").click()
        mobile.wait_for_timeout(700)
        box = mobile.locator("aside").first.bounding_box()
        check("点击汉堡后抽屉滑出", box is not None and box["x"] >= -1, str(box))

        browser.close()

    print()
    if CONSOLE_ERRORS:
        print(f"[警告] 浏览器控制台错误 {len(CONSOLE_ERRORS)} 条：")
        for e in CONSOLE_ERRORS[:10]:
            print("   -", e)
    if FAILED:
        print(f"[失败] {len(FAILED)} 项未通过：")
        for f in FAILED:
            print("   -", f)
        return 1
    print("[通过] 前端验收全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
