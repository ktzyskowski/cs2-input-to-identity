import os
from multiprocessing import Pool, cpu_count

import polars as pl
from demoparser2 import DemoParser

from collection.parser.abstract_parser import AbstractParser
from collection.parser.segment_parser import mouse_features
from collection.parser.segment_parser import key_features
from collection.parser.segment_parser.util import extract_tick_df, segment_player_df


class SegmentParser(AbstractParser):
    def __init__(self, directory: str, segment_length: int = 10, tickrate: int = 64, map_filter: list[str] = None):
        """Construct a new segment parser.

        :param directory: directory where samples are stored.
        :param segment_length: max length of each segment, in seconds
        :param tickrate: demo tickrate, default 64Hz.
        :param map_filter: a filter of map names to keep. This parser will skip processing maps if they are not played
                           on a map in this list.
        """
        super().__init__(directory)
        self._tickrate = tickrate
        self._segment_length = segment_length * self._tickrate
        self._map_filter = map_filter

    def parse_demo(self, path: str, match_id: str, map_id: int):
        # create demo parser and extract relevant tick information
        demo_parser = DemoParser(path)

        # check if demo map is in list to parse.
        map_name = demo_parser.parse_header().get("map_name", "NULL")
        if self._map_filter and map_name not in self._map_filter:
            return

        tick_df = extract_tick_df(demo_parser)

        # efficiently distribute work across multiple CPU cores
        with Pool(cpu_count()) as pool:
            steamids, player_dfs = zip(*tick_df.group_by("steamid"))
            feature_dfs = pool.map(self._parse_player, player_dfs)

        for (steamid,), feature_df in zip(steamids, feature_dfs):
            print(steamid, feature_df.shape)
            self._save_features(feature_df, match_id, map_id, steamid)

        # player_dfs = [player_df for _, player_df in tick_df.group_by("steamid")]
        # self._parse_player(player_dfs[0])

    def _parse_player(self, player_df: pl.DataFrame):
        """Parse a player DataFrame.

        :param player_df: the player DataFrame.
        :return:
        """
        # separate player trajectory into k-second segments
        segmented_player_df = segment_player_df(player_df, self._segment_length)

        # add team numbers to dataframe (differentiate between T/CT)
        team_numbers = segmented_player_df.unique(subset=["segment_id", "team_num"]).select(["segment_id", "team_num"])

        mouse_df = mouse_features.extract(segmented_player_df)
        key_df = key_features.extract(segmented_player_df)
        features = (
            mouse_df
            .join(key_df, on="segment_id", how="inner")
            .join(team_numbers, on="segment_id", how="inner")
            .sort("segment_id")
        )
        return features

    def _save_features(self, features: pl.DataFrame, match_id, map_id, player_id):
        filename = f"{match_id}/{map_id}/{player_id}.csv"
        path = os.path.join(self._directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        features.write_csv(path)


# TODO: TEST, REMOVE!
if __name__ == "__main__":
    demo_path = "/Users/ktz/msai/msthesis/res/blast-premier-fall-final-2024-g2-vs-spirit-bo3-keEog6FzQxxIbzN28Nh3S0/g2-vs-spirit-m1-dust2.dem"
    segment_parser = SegmentParser("")
    segment_parser.parse_demo(demo_path, "test-id", 0)
