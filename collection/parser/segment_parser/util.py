import pandas as pd
import polars as pl
from demoparser2 import DemoParser

from collection.parser.segment_parser.constants import KEY_FEATURES, MOUSE_FEATURES


def extract_tick_df(demo_parser: DemoParser) -> pl.DataFrame:
    """Extract raw mouse and key dynamics for every player, for every tick in parsed demo.

    Ticks when players are dead or are in warmup are filtered out.

    :param demo_parser: demo parser object.
    :return: raw mouse and key dataframe.
    """
    tick_df = demo_parser.parse_ticks([
        *KEY_FEATURES,
        *MOUSE_FEATURES,
        "is_alive",
        "team_num"
    ])
    tick_df[KEY_FEATURES] = tick_df[KEY_FEATURES].astype(int)
    tick_df = tick_df[tick_df.is_alive]
    tick_df = tick_df.drop(columns=["is_alive"])

    # get ticks for each round start event
    round_start_df = (
        demo_parser.parse_event("round_start")
        # in event a round starts multiple times (possibly due to restart?) keep most recent start
        .drop_duplicates(subset="round", keep="last")
    )
    # assign round numbers to each tick according to extracted start ticks
    tick_df = (
        pd.merge_asof(
            tick_df.sort_values("tick"),
            round_start_df.sort_values("tick"),
            left_on="tick", right_on="tick",
            direction="backward"
        )
        .dropna(subset=["round"])
    )
    tick_df["round"] = tick_df["round"].astype(int)

    # demoparser2 gives us pandas dataframes, convert pandas to polars for faster processing downstream
    tick_df = pl.from_pandas(tick_df)
    return tick_df


def segment_player_df(player_df: pl.DataFrame, segment_length: int) -> pl.DataFrame:
    """Given the tick data for a player, segment it by tick length and round.

    No segment shall be longer than segment_length, and a new segment is always created at the start of a new round,
    regardless of the length of the segment at the end of the prior round.

    :param player_df: the player tick information.
    :param segment_length: the segment length, in ticks.
    :return: a copy of the player DataFrame object, with a new column titled ``segment_id``.
    """
    return (
        player_df
        .with_columns(
            # subtract min from each round to make ticks start at 0
            (pl.col("tick") - pl.col("tick").min().over("round"))
            .alias("tick")
        )
        .with_columns(
            # calculate segment ID within each round
            ((pl.col("tick") // segment_length).cast(pl.Int32))
            .alias("segment_id")
        )
        .with_columns(
            # combine with round number to create unique segment IDs
            (pl.col("round") * 1_000 + pl.col("segment_id"))
            .alias("segment_id")
        )
        .sort(["round", "tick"])
    )


def fill_segment_ids(df: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Fill in all segments in the DataFrame with 0 if features could not be extracted.

    This occurs when a segment has no movements, so the row is omitted from feature calculations.

    :param df: the data.
    :param features: DataFrame with columns containing extracted features.
    :return: DataFrame with filled zeros for null values.
    """
    return (
        df
        .select("segment_id")
        .unique()
        .join(
            features,
            on="segment_id",
            how="left",
        )
        .sort("segment_id")
        .fill_null(0)
    )
