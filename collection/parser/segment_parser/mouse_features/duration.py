import polars as pl

from collection.parser.segment_parser.util import fill_segment_ids


def extract_duration(mouse_df: pl.DataFrame) -> pl.DataFrame:
    """

    :param mouse_df:
    :return:
    """
    features = (
        mouse_df
        # remove null rows (when mouse is not moving!)
        .filter(
            pl.col("mouse_movement_id").is_not_null()
        )
        # create unique movement ID across all segments
        .group_by("mouse_movement_id")
        .agg([
            pl.count().alias("duration") / 64,  # 64Hz => 64 ticks per second
            pl.first("segment_id")  # preserve segment_id for grouping
        ])
        .group_by("segment_id")
        .agg([
            pl.mean("duration").alias("mean_duration"),
            pl.std("duration").alias("std_duration"),
            pl.min("duration").alias("min_duration"),
            pl.max("duration").alias("max_duration"),
            pl.sum("duration").alias("sum_duration"),
        ])
    )
    return fill_segment_ids(mouse_df, features)
