import polars as pl

from collection.parser.segment_parser.util import fill_segment_ids


def extract_speed(mouse_df: pl.DataFrame) -> pl.DataFrame:
    features = ["yaw_speed", "pitch_speed", "yaw_speed_acc", "pitch_speed_acc"]
    features = (
        mouse_df
        .with_columns([
            pl.col("yaw_delta").abs().alias("yaw_speed"),
            pl.col("pitch_delta").abs().alias("pitch_speed"),
        ])
        .with_columns([
            pl.col("yaw_speed").diff().fill_null(0).alias("yaw_speed_acc"),
            pl.col("pitch_speed").diff().fill_null(0).alias("pitch_speed_acc"),
        ])
        .group_by("mouse_movement_id")
        .agg([
            *[pl.mean(feature).alias(f"mean_{feature}") for feature in features],
            *[pl.std(feature).alias(f"std_{feature}") for feature in features],
            *[pl.min(feature).alias(f"min_{feature}") for feature in features],
            *[pl.max(feature).alias(f"max_{feature}") for feature in features],
            pl.first("segment_id")  # preserve segment_id for grouping
        ])
        .group_by("segment_id")
        .agg([
            # aggregate previously computed statistics for each segment
            *[pl.mean(f"mean_{feature}") for feature in features],
            *[pl.std(f"std_{feature}") for feature in features],
            *[pl.min(f"min_{feature}") for feature in features],
            *[pl.max(f"max_{feature}") for feature in features]
        ])
    )
    return fill_segment_ids(mouse_df, features)
