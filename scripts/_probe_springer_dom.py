"""
_probe_springer_dom.py
~~~~~~~~~~~~~~~~~~~~~~
Opens one Springer issue page in a visible browser and prints all <a> hrefs
that contain '/article/' so we can identify the correct CSS selectors.
Run once to fix download_editorials.py selectors.
"""
import asyncio
from playwright.async_api import async_playwright

ISSUE_URL = 'https://link.springer.com/journal/11412/11/1'

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        print(f"Loading: {ISSUE_URL}")
        await page.goto(ISSUE_URL, wait_until='networkidle', timeout=60_000)
        await page.wait_for_timeout(3000)

        # Dismiss cookie banner
        try:
            btn = page.locator('button:has-text("Accept all cookies")')
            await btn.wait_for(state='visible', timeout=4000)
            await btn.click()
            await page.wait_for_timeout(1000)
        except Exception:
            pass

        # Wait for article links
        await page.wait_for_timeout(3000)

        # Dump all hrefs containing /article/
        hrefs = await page.eval_on_selector_all(
            'a[href]',
            'els => els.map(el => ({href: el.href, text: el.innerText.trim().substring(0,80), '
            'classes: el.className, parent: el.parentElement ? el.parentElement.tagName + "." + el.parentElement.className.substring(0,50) : ""}))'
            '.filter(x => x.href.includes("/article/") || x.href.includes("s11412"))'
        )

        print(f"\nFound {len(hrefs)} article links:\n")
        for item in hrefs[:20]:  # show first 20
            print(f"  href   : {item['href']}")
            print(f"  text   : {item['text'][:70]}")
            print(f"  classes: {item['classes'][:80]}")
            print(f"  parent : {item['parent'][:80]}")
            print()

        # Also dump a snippet of the page HTML to see structure
        html_snippet = await page.evaluate(
            '() => document.body.innerHTML.substring(0, 5000)'
        )
        print("\n--- PAGE HTML SNIPPET (first 5000 chars) ---")
        print(html_snippet[:5000])

        await browser.close()

asyncio.run(main())
