"""
download_editorials_2006_2015.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Downloads the FIRST article PDF from each issue of ijCSCL
(Springer journal 11412) for years 2006–2015 (Vol 1–10).

Issue count: 39
  Vol 1  (2006): 4 issues
  Vol 2  (2007): 3 issue-groups (issue 2-3 is a combined issue → handled via
                 DOI dedup: whichever of /2/2 or /2/3 is returned first wins;
                 the duplicate is skipped)
  Vol 3–10 (2008–2015): 4 issues each
  Total: 4 + 3 + 8×4 = 39

Skip-if-exists: if ijcscl_{year}_v{vol:02d}_i{issue}_*.pdf already in
  editorials/, the download is skipped.
DOI dedup: if a DOI was already downloaded in this run, the issue is skipped.

Run:  python scripts/download_editorials_2006_2015.py

Author  : Gerry Stahl
Created : March 5, 2026
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR          = Path('/Users/GStahl2/AI/elibrary/editorials')
BASE_URL            = 'https://link.springer.com'
JOURNAL             = '11412'
VOL_FIRST_YEAR      = 2006   # Vol 1 = 2006
YEAR_START          = 2006
YEAR_END            = 2015
LOGIN_PAUSE_SECONDS = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_filename(text: str, maxlen: int = 70) -> str:
    text = re.sub(r'[^\w\s\-]', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text[:maxlen]


def doi_from_article_url(url: str) -> str | None:
    m = re.search(r'(10\.\d{4,}/\S+)', url)
    return m.group(1).rstrip('.,;') if m else None


def pdf_url_from_doi(doi: str) -> str:
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
    """Return (article_url, article_title) for the first article on an issue page."""
    await page.goto(issue_url, wait_until='domcontentloaded', timeout=40_000)
    await page.wait_for_timeout(2500)
    await dismiss_cookie_banner(page)

    selectors = [
        ('h2.app-card-open__heading a',          True),
        ('h3.app-card-open__heading a',          True),
        ('li.app-article-list-row__item h3 a',   True),
        ('article a[href*="/article/"]',         False),
        ('a[href*="/article/10.1007/s11412"]',   False),
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


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    from playwright.async_api import async_playwright

    downloaded: list[tuple[str, str]] = []
    failed:     list[tuple[str, str]] = []
    seen_dois:  set[str]              = set()   # DOI dedup for combined issues

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        context = await browser.new_context(accept_downloads=True)
        page    = await context.new_page()

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

        for year in range(YEAR_START, YEAR_END + 1):
            vol = year - VOL_FIRST_YEAR + 1    # 2006 → 1, 2015 → 10

            for issue in range(1, 5):
                label     = f'Vol {vol} ({year}) Issue {issue}'
                issue_url = f'{BASE_URL}/journal/{JOURNAL}/{vol}/{issue}'

                # ── Skip if file already exists on disk ───────────────────
                existing = list(OUTPUT_DIR.glob(f'ijcscl_{year}_v{vol:02d}_i{issue}_*.pdf'))
                if existing:
                    print(f'\n[{label}]  ↷ Already downloaded: {existing[0].name}')
                    doi = doi_from_article_url(existing[0].stem)
                    if doi:
                        seen_dois.add(doi)
                    downloaded.append((label, existing[0].name))
                    continue

                print(f'\n[{label}]  {issue_url}')

                try:
                    article_url, title = await get_first_article(page, issue_url)
                except Exception as exc:
                    print(f'  ✗ Issue page error: {exc}')
                    failed.append((label, f'Issue page error: {exc}'))
                    continue

                if not article_url:
                    print('  ✗ No first article found — skipping (likely empty/combined issue)')
                    failed.append((label, 'No article link found'))
                    continue

                doi = doi_from_article_url(article_url)

                # ── DOI dedup: skip if this is a combined-issue duplicate ──
                if doi and doi in seen_dois:
                    print(f'  ↷ DOI already downloaded ({doi}) — combined issue, skipping')
                    downloaded.append((label, f'(duplicate of combined issue, DOI={doi})'))
                    continue

                print(f'  Article : {article_url}')
                if title:
                    print(f'  Title   : {title[:75]}')

                if not doi:
                    print('  ✗ Could not extract DOI from article URL')
                    failed.append((label, 'DOI extraction failed'))
                    continue

                pdf_url   = pdf_url_from_doi(doi)
                safe_ttl  = safe_filename(title) if title else f'v{vol:02d}_i{issue}'
                filename  = f'ijcscl_{year}_v{vol:02d}_i{issue}_{safe_ttl}.pdf'
                save_path = OUTPUT_DIR / filename
                print(f'  PDF URL : {pdf_url}')
                print(f'  Saving  : {filename}')

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
                            seen_dois.add(doi)
                            print(f'  ✓ Saved ({len(body)//1024} KB)')
                            downloaded.append((label, filename))
                        else:
                            print('  ✗ Response is not a PDF (paywall/login redirect)')
                            failed.append((label, 'Paywall — not a PDF response'))
                    else:
                        print(f'  ✗ HTTP {response.status} for PDF URL')
                        failed.append((label, f'HTTP {response.status}'))
                except Exception as exc:
                    print(f'  ✗ Download error: {exc}')
                    failed.append((label, f'Download error: {exc}'))

                await asyncio.sleep(1.2)

        await browser.close()

    # ── Final report ─────────────────────────────────────────────────────────
    real_downloads = [(l, f) for l, f in downloaded if not f.startswith('(duplicate')]
    print(f'\n{"="*62}')
    print(f'  DOWNLOAD COMPLETE — {len(real_downloads)} PDFs saved')
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
