import logging
import os
import sys
from tempfile import TemporaryDirectory

from collection.parser.segment_parser.parser import SegmentParser
from parser.abstract_parser import AbstractParser
from scraper.hltv_scraper import HltvScraper


class DataPipeline:
    """Main data pipeline for downloading and preprocessing HLTV demo data.

    This pipeline produces .npy files containing raw player key/mouse information for downstream ML tasks.
    """

    def __init__(self, res: str, scraper: HltvScraper, parser: AbstractParser):
        self._resource_directory = os.path.abspath(res)
        self._scraper = scraper
        self._parser = parser

    def get_match_hrefs(self) -> list[str]:
        """Get the match hrefs.

        These will be used to discover and scrape demo hrefs.
        If they have been scraped, load them from a file, otherwise scrape them from the HLTV website.

        :return: the match hrefs.
        """
        path = os.path.join(self._resource_directory, "match_hrefs.txt")

        # if file already exists, read and return data
        if os.path.exists(path):
            print("match_hrefs.txt exists...")
            with open(path, "r") as f:
                hrefs = [line.rstrip() for line in f.readlines()]
                return hrefs

        # otherwise we need to create the file first
        os.makedirs(self._resource_directory, exist_ok=True)
        hrefs = self._scraper.scrape_match_hrefs()
        with open(path, "w") as f:
            for href in hrefs:
                f.write(f"{href}\n")
        return hrefs

    def get_demo_hrefs(self, match_hrefs: list[str]) -> list[str]:
        """Get the demo hrefs.

        These will be used to download the match .dem files which contain the telemetry data.
        If they have been scraped, load them from a file, otherwise scrape them from the HLTV website.

        :return: the match hrefs.
        """
        path = os.path.join(self._resource_directory, "demo_hrefs.txt")
        if os.path.exists(path):
            print("demo_hrefs.txt exists...")
            with open(path, "r") as f:
                hrefs = [line.rstrip() for line in f.readlines()]
                return hrefs

        hrefs = self._scraper.scrape_demo_hrefs(match_hrefs)
        with open(path, "w") as f:
            for href in hrefs:
                f.write(f"{href}\n")
        return hrefs

    def download_demos(self, demo_hrefs: list[str]):
        """Download the .dem files.

        The scraper downloads RAR archives from the HLTV website containing the .dem files.
        This method downloads the RAR archive to a temporary directory, extracts the file contents,
        and then parses the file into a parseable format and saves it in the resources directory.
        This enables storage of only the data we need without the overhead of storing large .dem files.

        :param demo_hrefs: the demo hrefs.
        """
        demo_directory = os.path.join(self._resource_directory, "demo")
        if not os.path.exists(demo_directory):
            os.makedirs(demo_directory)

        for idx, demo_href in enumerate(demo_hrefs):
            match_id = demo_href.split("/")[-1]
            if self._parser.parsed(match_id):
                # already attempted to download this href, continue on to next
                logging.info(f"skipping demo {match_id}...")
                continue

            logging.info(f"{idx}\tdownloading demo {match_id}...")

            # create temp directory to hold working files
            with TemporaryDirectory() as tmpdir:
                self.download_demo(match_id, demo_href, tmpdir)

            # create flag to indicate that this demo has been downloaded
            self._parser.mark_parsed(match_id)

    def download_demo(self, match_id: str, demo_href: str, directory: str):
        """Download a match, and parse the demos contained within.

        :param match_id: the ID of the match.
        :param demo_href: the href of the demo on HLTV.
        :param directory: the directory to download the demo to.
]        """
        # 1. scrape demo and extract archive to directory
        self._scraper.scrape_demos(demo_href, directory)
        logging.debug("scraped demos")

        # 2. parse demos within directory, and do whatever the parser does with them
        # (most likely implementation is to apply some transformation to the .dem file
        #  and save it to the /res directory)
        self._parser.parse_directory(directory, match_id=match_id)
        logging.debug("parsed demos")

    def run(self):
        """Run the data collection pipeline."""
        # 1. get match hrefs
        match_hrefs = self.get_match_hrefs()
        logging.info("match hrefs collected")

        # 2. get demo hrefs
        demo_hrefs = self.get_demo_hrefs(match_hrefs)
        logging.info("demo hrefs collected")

        # 3. download demos and process into npy arrays
        self.download_demos(demo_hrefs)


def main():
    """Main function.

    Instantiates scraper, parser, and data collection pipeline. It then runs the pipeline.
    """

    scraper = HltvScraper(headless=False)
    parser = SegmentParser(directory="res/mnk5-dust2", segment_length=5, map_filter=["de_dust2"])
    pipeline = DataPipeline("res/", scraper=scraper, parser=parser)
    pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    main()
