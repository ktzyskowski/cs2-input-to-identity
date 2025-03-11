import polars as pl

from collection.parser.segment_parser.constants import KEY_FEATURES
from collection.parser.segment_parser.key_features.entropy import extract_entropy
from collection.parser.segment_parser.key_features.n_key_presses import extract_n_key_presses
from collection.parser.segment_parser.key_features.n_keys_down import extract_n_keys_down
from collection.parser.segment_parser.util import fill_segment_ids


def extract(segmented_player_df: pl.DataFrame) -> pl.DataFrame:
    segmented_player_df = extract_key_ids(segmented_player_df)

    n_keys_df = extract_n_keys_down(segmented_player_df)
    entropy_df = extract_entropy(segmented_player_df)
    n_key_presses_df = extract_n_key_presses(segmented_player_df)
    key_transition_time_df = extract_key_transition_time(segmented_player_df)
    key_down_time_df = extract_key_down_time(segmented_player_df)

    features = (
        n_keys_df
        .join(n_key_presses_df, on="segment_id", how="inner")
        .join(key_transition_time_df, on="segment_id", how="inner")
        .join(key_down_time_df, on="segment_id", how="inner")
        .with_columns(entropy_df)
    )

    return features


def extract_key_ids(segmented_player_df: pl.DataFrame) -> pl.DataFrame:
    features = (
        segmented_player_df
        # detect when a button is pressed
        .with_columns([
            ((pl.col(key).shift().fill_null(0) == 0) & (pl.col(key) == 1)).alias(f"{key}_press")
            for key in KEY_FEATURES
        ])
        # assign unique button press ID
        .with_columns([
            pl.cum_sum(f"{key}_press").alias(f"{key}_press_id")
            for key in KEY_FEATURES
        ])
        # assign null to moments button is not pressed
        .with_columns([
            pl.when(pl.col(key) == 1).then(pl.col(f"{key}_press_id")).otherwise(None)
            for key in KEY_FEATURES
        ])
    )
    return features


def extract_key_transition_time(segmented_player_df: pl.DataFrame) -> pl.DataFrame:
    features = None
    for key in KEY_FEATURES:
        key_transitions = (
            segmented_player_df
            .with_columns(pl.arange(0, pl.len()).alias("index"))  # preserve order
            .group_by(["segment_id", f"{key}_press_id"])
            .agg([
                pl.min("index").alias("start_time"),
                pl.max("index").alias("end_time"),
            ])
            .drop_nulls()
            .sort(["segment_id", "start_time"])
            .with_columns((pl.col("start_time").shift(-1) - pl.col("end_time")).alias(f"{key}_transition_time"))
            .filter(pl.col(f"{key}_transition_time") >= 0)  # remove negative/invalid transitions
        )
        key_transition_stats = (
            key_transitions
            .group_by("segment_id")
            .agg([
                pl.min(f"{key}_transition_time").alias(f"min_{key}_transition"),
                pl.max(f"{key}_transition_time").alias(f"max_{key}_transition"),
                pl.mean(f"{key}_transition_time").alias(f"mean_{key}_transition"),
                pl.std(f"{key}_transition_time").alias(f"std_{key}_transition"),
                pl.count(f"{key}_transition_time").alias(f"sum_{key}_transition"),
            ])
        )
        key_stats = fill_segment_ids(segmented_player_df, key_transition_stats)
        features = key_stats if features is None else features.with_columns(key_stats)
    return features


def extract_key_down_time(segmented_player_df: pl.DataFrame) -> pl.DataFrame:
    features = None
    for key in KEY_FEATURES:
        key_durations = (
            segmented_player_df
            .group_by(["segment_id", f"{key}_press_id"])  # Group by segment and button press ID
            .agg(pl.count().alias(f"{key}_duration"))
            .drop_nulls()  # Remove invalid press IDs
        )
        key_stats = (
            key_durations
            .group_by("segment_id")
            .agg([
                pl.min(f"{key}_duration").alias(f"min_{key}_duration"),
                pl.max(f"{key}_duration").alias(f"max_{key}_duration"),
                pl.mean(f"{key}_duration").alias(f"mean_{key}_duration"),
                pl.std(f"{key}_duration").alias(f"std_{key}_duration"),
                pl.count(f"{key}_duration").alias(f"sum_{key}_duration"),
            ])
        )
        key_stats = fill_segment_ids(segmented_player_df, key_stats)
        features = key_stats if features is None else features.with_columns(key_stats)
    return features
