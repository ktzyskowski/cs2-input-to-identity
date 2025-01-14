import os.path
from tempfile import TemporaryDirectory

import patoolib
import stealth_requests as requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from scraper.urls import ResultsUrl


class HltvScraper:
    def __init__(self, res_dir: str):
        self.res_dir = res_dir

    def _scrape_match_hrefs(self):
        hrefs = []
        # 1574 hardcoded in
        for offset in range(0, 1574, 100):
            page_html = self._download_html(str(ResultsUrl(offset=offset)))
            page_hrefs = self._match_hrefs_from_html(page_html)
            hrefs.extend(page_hrefs)
        return hrefs

    def _scrape_demo_hrefs(self, match_hrefs: list[str]):
        hrefs = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            for match_href in match_hrefs:
                context = browser.new_context()
                page = context.new_page()
                page.goto("https://www.hltv.org" + match_href)
                page.get_by_text("Allow all cookies").click()
                html = page.content()
                page.close()
                soup = BeautifulSoup(html, "html.parser")
                for a_tag in soup.find_all("a", {"href": True}):
                    if a_tag["href"].startswith("/download/demo"):
                        print(a_tag["href"])
                        hrefs.append(a_tag["href"])
        return hrefs

    def _download_html(self, url: str):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url)

            page.get_by_text("Allow all cookies").click()
            return page.content()

    def _match_hrefs_from_html(self, html: str):
        hrefs = []
        soup = BeautifulSoup(html, "html.parser")

        if results_div := soup.find("div", {"class": "results"}):
            for a_tag in results_div.find_all("a", {"href": True}):
                if a_tag["href"].startswith("/matches"):
                    hrefs.append(a_tag["href"])
            return hrefs
        else:
            raise ValueError("results not found")

    def get_match_hrefs(self):
        path = os.path.join(self.res_dir, "match_hrefs.txt")
        if not os.path.exists(path):
            hrefs = self._scrape_match_hrefs()
            with open(path, "w") as f:
                for href in hrefs:
                    f.write(f"{href}\n")
        else:
            with open(path, "r") as f:
                hrefs = [line.rstrip() for line in f.readlines()]
        return hrefs

    def get_demo_hrefs(self, match_hrefs: list[str]):
        path = os.path.join(self.res_dir, "demo_hrefs.txt")
        if not os.path.exists(path):
            demo_hrefs = self._scrape_demo_hrefs(match_hrefs)
            with open(path, "w") as f:
                for href in demo_hrefs:
                    f.write(f"{href}\n")
        else:
            with open(path, "r") as f:
                hrefs = [line.rstrip() for line in f.readlines()]
        return hrefs

    def download_demo(self, demo_href: str):
        # local_path = os.path.join(self.res_dir, "demo", local_filename)
        url = "https://www.hltv.org" + demo_href
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with TemporaryDirectory() as tmpdir:
            # write rar demo to file inside temp dir
            archive_name = url.split('/')[-1] + ".rar"
            archive_path = os.path.join(tmpdir, archive_name)
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content():
                    f.write(chunk)

            # extract rar to res dir
            demo_id = demo_href.split("/")[-1]
            patoolib.extract_archive(archive_path, outdir=os.path.join(self.res_dir, "demo", demo_id))

        # flatten directory, if necessary
        outer_dir = os.path.join(self.res_dir, "demo", demo_id)
        inner_dir = os.path.join(outer_dir, demo_id)
        if os.path.exists(inner_dir):
            for dem_file in os.listdir(inner_dir):
                dem_path = os.path.join(inner_dir, dem_file)
                os.rename(dem_path, os.path.join(outer_dir, dem_file))
            os.rmdir(inner_dir)


def main():
    scraper = HltvScraper("../res")
    match_hrefs = scraper.get_match_hrefs()
    demo_hrefs = scraper.get_demo_hrefs(match_hrefs)

    for demo_href in demo_hrefs:
        demo_id = demo_href.split("/")[-1]
        if not os.path.exists(os.path.join("../res/demo", demo_id)):
            print(f"downloading {demo_id}...")
            scraper.download_demo(demo_href)

    # with ThreadPoolExecutor(max_workers=10) as executor, sync_playwright() as playwright:
    #     # with sync_playwright() as playwright:
    #     browser = playwright.chromium.launch(headless=False)
    #     fn = functools.partial(get_demo_href, browser=browser)
    #     demo_hrefs = list(executor.map(fn, hrefs))
    #
    # # for demo_href in demo_hrefs:
    # # print(demo_href)


if __name__ == "__main__":
    main()
