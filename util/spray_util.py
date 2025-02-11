import os

import numpy as np
import pandas as pd


def load_spray_df(directory: str):
    data = []
    for match_id in os.listdir(directory):
        if not match_id.isnumeric():
            continue
        match_directory = os.path.join(directory, match_id)
        for map_id in os.listdir(match_directory):
            if not map_id.isnumeric():
                continue
            map_directory = os.path.join(match_directory, map_id)
            for player_id in os.listdir(map_directory):
                if not player_id.isnumeric():
                    continue
                player_directory = os.path.join(map_directory, player_id)
                for filename in os.listdir(player_directory):
                    if not filename.endswith(".npy"):
                        continue
                    data.append([match_id, map_id, player_id, filename])

    df = pd.DataFrame(data, columns=["match_id", "map_id", "player_id", "filename"])
    lengths = []
    for idx, row in df.iterrows():
        filename = f"{row.match_id}/{row.map_id}/{row.player_id}/{row.filename}"
        path = os.path.join(directory, filename)
        array = np.load(path)
        lengths.append(array.shape[0])
    df["length"] = lengths
    return df
