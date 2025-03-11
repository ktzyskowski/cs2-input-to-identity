import polars as pl

from collection.parser.segment_parser.util import fill_segment_ids


def extract_total_distance(mouse_df: pl.DataFrame) -> pl.DataFrame:
    """Compute statistical averages over the total distance travelled per mouse movement in the given mouse data.

    The distance is defined as the sum of distances between each yaw/pitch point in the movement.
    This function extracts the mean, the standard deviation, the minimum, and maximum values for all mouse movements in
    each segment.

    :param mouse_df: the mouse data.
    :return: DataFrame with appended columns containing total distance travelled.
    """
    features = (
        mouse_df
        .group_by("mouse_movement_id")
        .agg(
            pl.sum("angular_displacement").alias("total_distance"),
            pl.first("segment_id")  # preserve segment_id for grouping
        )
        .group_by("segment_id")
        .agg(
            pl.mean("total_distance").alias("mean_total_distance"),
            pl.std("total_distance").alias("std_total_distance"),
            pl.min("total_distance").alias("min_total_distance"),
            pl.max("total_distance").alias("max_total_distance"),
            pl.sum("total_distance").alias("sum_total_distance"),
        )
    )
    return fill_segment_ids(mouse_df, features)
