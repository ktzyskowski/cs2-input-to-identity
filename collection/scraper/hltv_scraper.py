import logging
import os.path

import patoolib
import stealth_requests as requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from collection.scraper.urls import ResultsUrl


class HltvScraper:
    """HLTV data scraper.

    This object is used to collect demo files for downstream ML tasks.
    """

    def __init__(self, headless: bool = False):
        self.headless = headless

    def scrape_match_hrefs(self) -> list[str]:
        hrefs = []
        # 1574 hardcoded in
        for offset in range(0, 1574, 100):
            page_html = self._download_html(str(ResultsUrl(offset=offset)))
            page_hrefs = self._match_hrefs_from_html(page_html)
            hrefs.extend(page_hrefs)
        return hrefs

    def scrape_demo_hrefs(self, match_hrefs: list[str]) -> list[str]:
        hrefs = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            for match_href in match_hrefs:
                context = browser.new_context()
                page = context.new_page()
                page.goto("https://www.hltv.org" + match_href)
                # page.get_by_text("Allow all cookies").click()
                html = page.content()
                page.close()
                soup = BeautifulSoup(html, "html.parser")
                for a_tag in soup.find_all("a", {"href": True}):
                    if a_tag["href"].startswith("/download/demo"):
                        print(a_tag["href"])
                        hrefs.append(a_tag["href"])
        return hrefs

    def scrape_demos(self, demo_href: str, out: str) -> None:
        url = "https://www.hltv.org" + demo_href
        r = requests.get(url, stream=True)
        r.raise_for_status()

        logging.debug("request OK")

        # write RAR archive to a file inside given directory
        archive_name = url.split('/')[-1] + ".rar"
        archive_path = os.path.join(out, archive_name)
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content():
                f.write(chunk)

        logging.debug("archive downloaded")

        # extract RAR archive contents to same directory
        patoolib.extract_archive(archive_path, outdir=out, verbosity=-1)  # silence logs

        logging.debug("archive extracted")

    def _download_html(self, url: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url)

            page.get_by_text("Allow all cookies").click()
            return page.content()

    def _match_hrefs_from_html(self, html: str) -> list[str]:
        hrefs = []
        soup = BeautifulSoup(html, "html.parser")

        if results_div := soup.find("div", {"class": "results"}):
            for a_tag in results_div.find_all("a", {"href": True}):
                if a_tag["href"].startswith("/matches"):
                    hrefs.append(a_tag["href"])
            return hrefs
        else:
            raise ValueError("results not found")
