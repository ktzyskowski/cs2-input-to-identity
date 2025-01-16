import logging
import os

import pandas as pd
from demoparser2 import DemoParser


class NumPyParser:
    """A parser object that transforms demos into NumPy arrays."""

    def parse_demos(self, directory: str):
        """Parse the .dem files in the given directory, and yield the parsed samples.

        :param directory: the directory containing .dem files.
        :return: a generator over the processed .dem files, as NumPy arrays. The arrays are returned as a dictionary
        """
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".dem"):
                    path = os.path.join(root, file)
                    try:
                        yield self.load_samples(path)
                    except Exception:
                        logging.error(f"could not parse {path}")
                        continue

    def load_samples(self, demo_path: str):
        """Load and parse the samples from the .dem file at the given path.

        Each .dem file contains multiple rounds with multiple players.
        One sample is the entire round played by a player, so the return
        value of this method is a dictionary mapping player ID and round number
        to raw telemetry data as an array.

        :param demo_path: the .dem file path.
        :return: A dictionary containing the player telemetry data.
        """
        samples = {}
        df = self.parse_demo(demo_path)
        for (round_num, player_id), player_df in df.groupby(["total_rounds_played", "steamid"]):
            array = self.df_to_array(player_df)
            samples[f"{player_id}_{round_num}"] = array
        return samples

    def parse_demo(self, demo_path: str):
        """Parse the .dem file at the given path as a DataFrame.

        :param demo_path: the .dem file path.
        :return: a DataFrame containing the player telemetry data.
        """
        parser = DemoParser(demo_path)
        df = parser.parse_ticks(wanted_props=[
            "is_alive",
            "total_rounds_played",
            "FORWARD",
            "LEFT",
            "RIGHT",
            "BACK",
            "FIRE",
            "RIGHTCLICK",
            "RELOAD",
            "INSPECT",
            "USE",
            "ZOOM",
            "SCOREBOARD",
            "pitch",
            "yaw"
        ])
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(float)
        df["is_alive"] = df["is_alive"].astype(bool)
        return df

    def df_to_array(self, df: pd.DataFrame):
        """Convert the DataFrame into a NumPy array.

        :param df: the DataFrame.
        :return: an array.
        """
        df = df[df["is_alive"]]  # filter out data where the player is not alive, i.e. not playing.
        return df[
            [
                "FORWARD",
                "LEFT",
                "RIGHT",
                "BACK",
                "FIRE",
                "RIGHTCLICK",
                "RELOAD",
                "INSPECT",
                "USE",
                "ZOOM",
                "SCOREBOARD",
                "pitch",
                "yaw"
            ]
        ].to_numpy()
