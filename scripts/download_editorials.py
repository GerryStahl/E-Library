"""
download_editorials.py
~~~~~~~~~~~~~~~~~~~~~~
Downloads the FIRST article PDF from each issue of ijCSCL
(International Journal of Computer-Supported Collaborative Learning,
ISSN 1556-1607, Springer journal 11412) for years 2016–2025.

That is 10 years × 4 issues = 40 PDFs.

ijCSCL volume mapping (started 2006 = Vol 1):
  Vol 11 = 2016, Vol 12 = 2017, ..., Vol 20 = 2025

PDFs are saved to:  /Users/GStahl2/AI/elibrary/editorials/
Filename pattern:   ijcscl_YYYY_v{vol:02d}_i{issue}_{safe_title}.pdf

HOW IT WORKS
------------
1. A visible Chromium browser window opens and goes to the journal page.
2. If Springer asks you to log in, do so — the script pauses 30 s.
3. For each of the 40 issues the script:
      a. Loads the issue page, grabs the first article URL + title
         (selector confirmed: h2.app-card-open__heading a).
      b. Constructs the direct PDF URL from the article DOI.
      c. Downloads the PDF using the browser's authenticated session
         (page.request.get) and saves it to the editorials/ folder.

Run:  python scripts/download_editorials.py

Author  : Gerry Stahl
Created : March 5, 2026
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
OUTPUT_DIR          = Path('/Users/GStahl2/AI/elibrary/editorials')
BASE_URL            = 'https://link.springer.com'
JOURNAL             = '11412'
VOL_FIRST_YEAR      = 2006   # Vol 1 was 2006, so Vol 11 = 2016
YEAR_START          = 2016
YEAR_END            = 2025
LOGIN_PAUSE_SECONDS = 30     # seconds to let the user log in if prompted

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def safe_filename(text: str, maxlen: int = 70) -> str:
    """Return a filesystem-safe version of text."""
    text = re.sub(r'[^\w\s\-]', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text[:maxlen]


def doi_from_article_url(url: str) -> str | None:
    """Extract DOI like '10.1007/s11412-016-9230-x' from article URL."""
    m = re.search(r'(10\.\d{4,}/\S+)', url)
    return m.group(1).rstrip('.,;') if m else None


def pdf_url_from_doi(doi: str) -> str:
    """Construct the Springer direct-download PDF URL."""
    return f'{BASE_URL}/content/pdf/{doi}.pdf'


async def dismiss_cookie_banner(page) -> None:
    try:
        btn = page.locator('button:has-text("Accept all cookies")')
        await btn.wait_for(state='visible', timeout=4000)
        await btn.click()
        await page.wait_for_timeout(600)
    except Exception:
        pass


async def get_first_article(page, issue_url: str) -> tuple[str, str] | tuple[None, None]:
    """
    Navigate to an issue page and return (article_url, article_title)
    for the first article listed.

    Confirmed selector from DOM probe (March 2026):
        h2.app-card-open__heading a
    """
    await page.goto(issue_url, wait_until='domcontentloaded', timeout=40_000)
    await page.wait_for_timeout(2500)
    await dismiss_cookie_banner(page)

    # Ordered from most specific (confirmed) to broadest fallback
    selectors = [
        ('h2.app-card-open__heading a',              True),   # confirmed March 2026
        ('h3.app-card-open__heading a',              True),
        ('li.app-article-list-row__item h3 a',       True),
        ('article a[href*="/article/"]',             False),
        ('a[href*="/article/10.1007/s11412"]',       False),
    ]

    for sel, need_text in selectors:
        loc = page.locator(sel).first
        try:
            await loc.wait_for(state='visible', timeout=3000)
            href  = await loc.get_attribute('href')
            title = (await loc.inner_text()).strip() if need_text else ''
            if href:
                full_url = href if href.startswith('http') else BASE_URL + href
                return full_url, title
        except Exception:
            continue

    return None, None


# ── Main ───────────────────────────────────────────────────────────────────

async def main() -> None:
    from playwright.async_api import async_playwright

    downloaded: list[tuple[str, str]] = []
    failed:     list[tuple[str, str]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        context = await browser.new_context(accept_downloads=True)
        page    = await context.new_page()

        # ── Navigate to journal; give user time to log in ─────────────────
        print('\nOpening Springer journal page...')
        await page.goto(
            f'{BASE_URL}/journal/{JOURNAL}/volumes-and-issues',
            wait_until='domcontentloaded', timeout=40_000,
        )
        await dismiss_cookie_banner(page)
        print(
            f'\n{"="*62}\n'
            f'  Browser is open. If Springer asks you to log in, do so now.\n'
            f'  The script starts automatically in {LOGIN_PAUSE_SECONDS} s.\n'
            f'{"="*62}\n'
        )
        await asyncio.sleep(LOGIN_PAUSE_SECONDS)

        # ── Process all 40 issues ──────────────────────────────────────────
        for year in range(YEAR_START, YEAR_END + 1):
            vol = year - VOL_FIRST_YEAR + 1   # 2016 → 11, 2025 → 20

            for issue in range(1, 5):
                label     = f'Vol {vol} ({year}) Issue {issue}'
                issue_url = f'{BASE_URL}/journal/{JOURNAL}/{vol}/{issue}'

                # Skip if already downloaded (any filename matching this year/vol/issue)
                existing = list(OUTPUT_DIR.glob(f'ijcscl_{year}_v{vol:02d}_i{issue}_*.pdf'))
                if existing:
                    print(f'\n[{label}]  ↷ Already downloaded: {existing[0].name}')
                    downloaded.append((label, existing[0].name))
                    continue

                print(f'\n[{label}]  {issue_url}')

                # 1. Find the first article URL + title on the issue page
                try:
                    article_url, title = await get_first_article(page, issue_url)
                except Exception as exc:
                    print(f'  ✗ Issue page error: {exc}')
                    failed.append((label, f'Issue page error: {exc}'))
                    continue

                if not article_url:
                    print('  ✗ No first article found on issue page')
                    failed.append((label, 'No article link found'))
                    continue

                print(f'  Article : {article_url}')
                if title:
                    print(f'  Title   : {title[:75]}')

                # 2. Build PDF URL from DOI embedded in article URL
                doi = doi_from_article_url(article_url)
                if not doi:
                    print('  ✗ Could not extract DOI from article URL')
                    failed.append((label, 'DOI extraction failed'))
                    continue

                pdf_url  = pdf_url_from_doi(doi)
                safe_ttl = safe_filename(title) if title else f'v{vol:02d}_i{issue}'
                filename = f'ijcscl_{year}_v{vol:02d}_i{issue}_{safe_ttl}.pdf'
                save_path = OUTPUT_DIR / filename
                print(f'  PDF URL : {pdf_url}')
                print(f'  Saving  : {filename}')

                # 3. Download using the browser's authenticated session
                try:
                    response = await page.request.get(
                        pdf_url,
                        timeout=60_000,
                        headers={'Accept': 'application/pdf,*/*'},
                    )
                    if response.ok:
                        body = await response.body()
                        if body[:4] == b'%PDF':
                            save_path.write_bytes(body)
                            print(f'  ✓ Saved ({len(body)//1024} KB)')
                            downloaded.append((label, filename))
                        else:
                            # Not a real PDF — probably a login redirect page
                            print('  ✗ Response is not a PDF (paywall/login redirect)')
                            failed.append((label, 'Paywall — not a PDF response'))
                    else:
                        print(f'  ✗ HTTP {response.status} for PDF URL')
                        failed.append((label, f'HTTP {response.status}'))
                except Exception as exc:
                    print(f'  ✗ Download error: {exc}')
                    failed.append((label, f'Download error: {exc}'))

                await asyncio.sleep(1.2)   # be polite to the server

        await browser.close()

    # ── Final report ───────────────────────────────────────────────────────
    print(f'\n{"="*62}')
    print(f'  DOWNLOAD COMPLETE — {len(downloaded)}/40 succeeded')
    print(f'{"="*62}')

    if downloaded:
        print(f'\nDOWNLOADED ({len(downloaded)}):')
        for lbl, fname in downloaded:
            print(f'  ✓ {lbl}: {fname}')

    if failed:
        print(f'\nFAILED ({len(failed)}):')
        for lbl, reason in failed:
            print(f'  ✗ {lbl}: {reason}')

    print(f'\nFiles saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    asyncio.run(main())
