import polars as pl

from collection.parser.segment_parser.util import fill_segment_ids


def extract_straight_distance(mouse_df: pl.DataFrame) -> pl.DataFrame:
    """Compute averages over the Euclidean distance travelled per mouse movement in the given mouse data.

    The distance is defined as the difference between the start and end yaw/pitch points in the movement.
    This function extracts the mean, the standard deviation, the minimum, and maximum values for all mouse movements in
    each segment.

    :param mouse_df: the mouse data.
    :return: DataFrame with appended columns containing total distance travelled.
    """
    features = (
        mouse_df
        .group_by("mouse_movement_id")
        # collect start/end points
        .agg(
            pl.first("yaw").alias("yaw_start"),
            pl.first("pitch").alias("pitch_start"),
            pl.last("yaw").alias("yaw_end"),
            pl.last("pitch").alias("pitch_end"),
            pl.first("segment_id")  # preserve segment_id for grouping
        )
        # compute deltas
        .with_columns([
            (pl.col("yaw_end") - pl.col("yaw_start")).alias("yaw_delta"),
            (pl.col("pitch_end") - pl.col("pitch_start")).alias("pitch_delta"),
        ])
        # correct for yaw wrapping [-180, 179)
        .with_columns(
            ((pl.col("yaw_delta") + 180) % 360 - 180).alias("yaw_delta"),
        )
        # compute Euclidean (straight-line) distance
        .with_columns(
            (pl.col("yaw_delta").pow(2) + pl.col("pitch_delta").pow(2)).sqrt()
            .alias("straight_line_distance")
        )
        # average stats over segments
        .group_by("segment_id").agg(
            pl.mean("straight_line_distance").alias("mean_straight_distance"),
            pl.std("straight_line_distance").alias("std_straight_distance"),
            pl.min("straight_line_distance").alias("min_straight_distance"),
            pl.max("straight_line_distance").alias("max_straight_distance"),
            pl.sum("straight_line_distance").alias("sum_straight_distance"),
        )
    )
    return fill_segment_ids(mouse_df, features)
