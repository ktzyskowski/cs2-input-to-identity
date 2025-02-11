import os

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

from collection.parser.abstract_parser import AbstractParser


# save this code snippet potentially
# fill in missing values with interpolation
# weapon_fire_ticks = weapon_fire_ticks \
#     .set_index("tick") \
#     .reindex(range(start_tick, end_tick))[["pitch", "yaw"]] \
#     .interpolate() \
#     .bfill() \
#     .ffill()

class SprayParser(AbstractParser):
    """Spray pattern parser class.

    This parser extracts spray control samples from players within the demo, when they are spraying a weapon. A spray
    is a repeatable mouse movement in CS2, which gives hope that differences between players can lead to solid
    classification performance.
    """

    def __init__(self, directory: str, min_shots: int = 5, max_dist: float = 3.0, weapon: str = "weapon_ak47"):
        """Initialize the spray parser.

        :param directory: the resource directory where we should save parsed sprays.
        :param min_shots: the minimum number of shots that must be fired consecutively to be considered a spray.
        :param max_dist: the maximum deviation in degrees between shots in a spray. This argument is used to filter
                         against spray transfers and other scenarios where the spray control "breaks".
        :param weapon: the weapon to collect spray patterns for. Defaults to the AK-47.
        """
        super().__init__(directory)
        self.min_shots = min_shots
        self.max_dist = max_dist
        self.weapon = weapon

    def parse_demo(self, path: str, match_id: str, map_id: int):
        """Parse a demo file and save the sprays to the resource directory.

        :param path: the path to the .dem file.
        :param match_id: the match ID.
        :param map_id: the map ID.
        """
        # collect weapon_fire and mouse dataframes
        parser = DemoParser(path)
        weapon_fire_df = self._get_weapon_fire_df(parser)
        mouse_df = parser.parse_ticks(["pitch", "yaw"])

        # iterate through each spray
        for spray_id, spray_df in weapon_fire_df.groupby("spray_id"):
            if spray_df.shape[0] < self.min_shots:
                continue

            player_id = spray_df.iloc[0].user_steamid

            # spray data is N x (pitch, yaw) in degrees
            spray_data = self._link_dfs(mouse_df, spray_df, player_id)
            # translate spray so it originates from (0, 0)
            spray_data -= spray_data[0]

            # skip sprays that "jump" out of control (perhaps a spray transfer?)
            # computed Euclidean distance
            dist = np.abs(np.sqrt(np.sum(np.square(np.diff(spray_data, axis=0)), axis=1)))
            if np.any(dist > self.max_dist):
                continue

            self._save(spray_data, match_id, map_id, player_id, spray_id)

    def _get_weapon_fire_df(self, parser: DemoParser):
        """Given a demo parser, extract a DataFrame containing the weapon fire events.

        Extra processing is done to aggregate each weapon fire event into a collection of sprays. Each spray is given a
        unique numeric identifier in a new column.

        The weapon fire events are filtered by weapon type. The default is the AK47.

        :param parser: the demo parser.
        :return: a DataFrame containing the weapon fire events.
        """
        # slow calls at the start
        weapon_fire_df = parser.parse_event("weapon_fire")
        tick_df = parser.parse_ticks(["FIRE", "active_weapon_ammo"], ticks=weapon_fire_df.tick)

        # filter by specific weapon and sort by player ID and tick
        weapon_fire_df = weapon_fire_df[weapon_fire_df.weapon == self.weapon]
        weapon_fire_df = weapon_fire_df.sort_values(by=["user_steamid", "tick"])

        # collect spray IDs
        spray_ids = []
        current_spray_id = 0
        for _, weapon_fire_row in weapon_fire_df.iterrows():
            tick, steamid = weapon_fire_row.tick, int(weapon_fire_row.user_steamid)
            tick_row = tick_df[(tick_df.tick == tick) & (tick_df.steamid == steamid)]
            if not tick_row.iloc[0].FIRE:
                # if FIRE is not down, this weapon fire is start of a new spray
                current_spray_id = current_spray_id + 1
            spray_ids.append(current_spray_id)
        weapon_fire_df["spray_id"] = spray_ids

        weapon_fire_df = weapon_fire_df.reset_index(drop=True)
        return weapon_fire_df

    def _link_dfs(self, mouse_df: pd.DataFrame, spray_df: pd.DataFrame, player_id: str):
        """Link the mouse angles with a spray.

        :param mouse_df: the mouse angles DataFrame.
        :param spray_df: the spray DataFrame.
        :param player_id: the player ID.
        :return: a NumPy containing the pitch/yaw angles for every tick in the player's spray.
        """
        data_df = mouse_df[mouse_df.tick.isin(spray_df.tick.values)]
        data_df = data_df[data_df.steamid == int(player_id)]
        return data_df[["pitch", "yaw"]].to_numpy()

    def _save(self, array: np.ndarray, match_id: str, map_id: int, player_id: int, spray_id: int):
        """Save the spray to disk.

        :param array: the data.
        :param match_id: the match ID.
        :param map_id: the map ID (0, 1, 2, 3, or 4).
        :param player_id: the player ID.
        :param spray_id: the spray ID.
        """
        filename = f"{match_id}/{map_id}/{player_id}/{spray_id}.npy"
        path = os.path.join(self._directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, array)
