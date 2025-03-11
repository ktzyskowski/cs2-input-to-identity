import polars as pl

from collection.parser.segment_parser.constants import KEY_FEATURES


def extract_n_key_presses(segmented_player_df: pl.DataFrame) -> pl.DataFrame:
    features = (
        segmented_player_df
        .with_columns([
            ((pl.col(key).shift() == 0) & (pl.col(key) == 1)).alias(f"{key}_presses").fill_null(False).cast(pl.UInt32)
            for key in KEY_FEATURES
        ])
        .group_by("segment_id")
        .agg(
            pl.col([f"{key}_presses" for key in KEY_FEATURES]).sum(),
        )
        .sort("segment_id")
        .with_columns(
            pl.sum_horizontal([f"{key}_presses" for key in KEY_FEATURES]).alias("total_presses")
        )
    )
    return features
