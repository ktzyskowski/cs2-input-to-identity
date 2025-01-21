import logging
import os

import pandas as pd
from demoparser2 import DemoParser


class ArrayParser:
    def parse_demos(self, directory: str):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".dem"):
                    path = os.path.join(root, file)
                    try:
                        yield self.parse_demo(path)
                    except:
                        logging.error(f"could not parse {path}")

    def parse_demo(self, path: str):
        parser = DemoParser(path)
        df = parser.parse_ticks([
            "is_alive",
            "is_warmup_period",
            "is_freeze_period",
            "total_rounds_played",
            "FORWARD",
            "BACK",
            "LEFT",
            "RIGHT",
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
        float_cols = df.drop(
            ["is_alive", "is_warmup_period", "is_freeze_period", "total_rounds_played"],
            axis=1
        ).columns
        df[float_cols] = df[float_cols].astype(float)

        # filter out when players are dead
        df = df[df["is_alive"]].drop("is_alive", axis=1)
        # filter out warmup
        df = df[df["is_warmup_period"] == False].drop("is_warmup_period", axis=1)
        # filter out freeze time
        df = df[df["is_freeze_period"] == False].drop("is_freeze_period", axis=1)

        df = df.reset_index(drop=True)
        player_rounds = self.extract_player_rounds(df)
        return player_rounds

    def extract_player_rounds(self, df: pd.DataFrame):
        player_rounds = {}
        for (round_num, player_id), player_df in df.groupby(["total_rounds_played", "steamid"]):
            array = self._df_to_array(player_df)
            player_rounds[f"{player_id}_{round_num}"] = array
        return player_rounds

    def _df_to_array(self, df: pd.DataFrame):
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
