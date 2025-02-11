import logging
import os
from abc import ABC, abstractmethod


class AbstractParser(ABC):
    """Abstract parser class."""

    def __init__(self, directory: str):
        """Constructor.

        :param directory: directory where processed data and metadata are stored.
        """
        self._directory = os.path.abspath(directory)

    def parse_directory(self, directory: str, match_id: str):
        """Parse .dem files within the given directory.

        :param directory: the directory that contains all .dem files to parse. Traversed recursively.
        :param match_id: the match ID which is associated with the given .dem files.
        """
        map_id = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".dem"):
                    path = os.path.join(root, file)
                    try:
                        logging.info(f"parsing {path}")
                        self.parse_demo(path, match_id=match_id, map_id=map_id)
                        map_id += 1
                    except Exception as e:
                        logging.error(f"could not parse {path} - {e}")

    @abstractmethod
    def parse_demo(self, path: str, match_id: str, map_id: int):
        """Parse a single .dem file.

        :param path: the path of the .dem file to parse.
        :param match_id: the match ID for this .dem file.
        :param map_id: the generic map ID for this .dem file. Not guaranteed to align with order of maps played.
        """
        pass

    def parsed(self, match_id: str):
        """Return where the given match was parsed.

        :param match_id: the match ID.
        :return: True if the match was parsed already, False otherwise.
        """
        parsed_directory = os.path.join(self._directory, ".parsed")
        os.makedirs(parsed_directory, exist_ok=True)
        flagpath = os.path.join(parsed_directory, match_id)
        return os.path.exists(flagpath)

    def mark_parsed(self, match_id: str):
        """Mark this match as parsed.

        :param match_id: the match ID.
        """
        parsed_directory = os.path.join(self._directory, ".parsed")
        os.makedirs(parsed_directory, exist_ok=True)
        flagpath = os.path.join(parsed_directory, match_id)
        open(flagpath, "a").close()
