"""Probe article page for PDF button selector."""
import asyncio
from playwright.async_api import async_playwright

ARTICLE_URL = 'https://link.springer.com/article/10.1007/s11412-016-9230-x'

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(ARTICLE_URL, wait_until='networkidle', timeout=60_000)
        await page.wait_for_timeout(2000)
        try:
            btn = page.locator('button:has-text("Accept all cookies")')
            await btn.wait_for(state='visible', timeout=3000)
            await btn.click()
            await page.wait_for_timeout(800)
        except Exception:
            pass
        items = await page.eval_on_selector_all(
            'a, button',
            'els => els'
            '.filter(el => el.outerHTML.toLowerCase().includes("pdf") || el.innerText.toLowerCase().includes("pdf"))'
            '.map(el => ({tag:el.tagName, text:el.innerText.trim().substring(0,60),'
            ' href:el.href||"", classes:el.className.substring(0,60),'
            ' dataTest:el.getAttribute("data-test")||""}))'
        )
        print(f"\nPDF-related elements ({len(items)}):")
        for item in items[:10]:
            print(f"  <{item['tag']}> text='{item['text']}' href='{item['href'][:80]}' data-test='{item['dataTest']}' class='{item['classes']}'")
        await browser.close()

asyncio.run(main())
