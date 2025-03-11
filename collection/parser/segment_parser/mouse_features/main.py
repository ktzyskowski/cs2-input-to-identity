import polars as pl

from collection.parser.segment_parser.mouse_features.count import extract_count
from collection.parser.segment_parser.mouse_features.duration import extract_duration
from collection.parser.segment_parser.mouse_features.speed import extract_speed
from collection.parser.segment_parser.mouse_features.straight_distance import extract_straight_distance
from collection.parser.segment_parser.mouse_features.total_distance import extract_total_distance


def extract(segmented_player_df: pl.DataFrame) -> pl.DataFrame:
    """Extract mouse features from the segmented player DataFrame.

    :param segmented_player_df: player DataFrame, in segments.
    :return: mouse features for each segment.
    """
    mouse_df = segmented_player_df.select(["yaw", "pitch", "segment_id"])
    mouse_df = extract_velocity_and_acceleration(mouse_df)
    mouse_df = extract_mouse_movements(mouse_df)

    # extract stats on mouse movements
    count_df = extract_count(mouse_df)
    # (mean, std, min, max, sum)
    total_distance_df = extract_total_distance(mouse_df)
    straight_distance_df = extract_straight_distance(mouse_df)
    duration_df = extract_duration(mouse_df)
    speed_df = extract_speed(mouse_df)

    features = (
        count_df
        .join(total_distance_df, on="segment_id", how="inner")
        .join(straight_distance_df, on="segment_id", how="inner")
        .join(duration_df, on="segment_id", how="inner")
        .join(speed_df, on="segment_id", how="inner")
    )
    return features


def extract_velocity_and_acceleration(mouse_df: pl.DataFrame) -> pl.DataFrame:
    """Extract the angular velocity and acceleration from the given mouse data.

    :param mouse_df: DataFrame containing yaw, pitch at each tick.
    :return: DataFrame with appended columns containing velocity and acceleration.
    """

    df = (
        mouse_df
        .with_columns([
            pl.col("pitch").diff().fill_null(0).alias("pitch_delta"),
            pl.col("yaw").diff().fill_null(0).alias("yaw_delta"),
        ])
        .with_columns(
            # handle circular wrapping
            (((pl.col("yaw_delta") + 180) % 360) - 180).alias("yaw_delta"),
        )
        .with_columns(
            (pl.col("yaw_delta").pow(2) + pl.col("pitch_delta").pow(2)).sqrt()
            .alias("angular_displacement")
        )
    )
    return df


def extract_mouse_movements(mouse_df: pl.DataFrame, at_rest_threshold=0.01) -> pl.DataFrame:
    """Identify individual mouse movements within the given mouse data.

    Three new columns are appended to the returned dataframe:
        - `at_rest`: binary 1 or 0 if the mouse is at rest (computed from tangential velocity).
        - `movement_start`: binary 1 or 0 to detect when `at_rest` transitions from 1->0.
        - `mouse_movement_id`: assigns a unique identifier to each separate mouse movement.
                               ticks when mouse is at rest have `null` identifiers.

    :param mouse_df:
    :param at_rest_threshold: the maximum velocity for a mouse to be considered "at rest".
    :return: DataFrame with appended columns containing mouse movement identifier.
    """
    df = (
        mouse_df
        # append new column `at_rest` to see when mouse is not in motion
        .with_columns(
            (pl.col("angular_displacement") <= at_rest_threshold).cast(pl.Int32()).alias("at_rest")
        )
        # identify when `at_rest` transitions 1->0 (a new movement starts)
        .with_columns(
            ((pl.col("at_rest").shift(1).fill_null(1) == 0) & (pl.col("at_rest") == 1))
            .cast(pl.Int32())
            .alias("movement_start")
        )
        # create `mouse_movement_id` by cumulatively summing movement_start and filling forward
        .with_columns(
            pl.col("movement_start")
            .cum_sum()
            .over("segment_id")
            .alias("mouse_movement_id")
        )
        .with_columns(
            pl.when(pl.col("at_rest") == 0)
            .then(pl.col("mouse_movement_id"))
            .otherwise(None)
            .alias("mouse_movement_id")
        )
        # create global movement_id across segments
        .with_columns(
            (pl.col("segment_id").cast(pl.Utf8) + "_" + pl.col("mouse_movement_id").cast(pl.Utf8))
            .alias("mouse_movement_id")
        )
    )
    return df
