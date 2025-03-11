import polars as pl

from collection.parser.segment_parser.constants import KEY_FEATURES


def extract_n_keys_down(segmented_player_df: pl.DataFrame) -> pl.DataFrame:
    """Extract the distribution of time that N keys are being held down during each segment.

    For example, if no keys are held down during an entire segment, then the feature vector will be [1 0 0 0 ... k]
    for K keys. If most of the time players only hold 1-3 keys, then these will be the most populous indices.

    :param segmented_player_df:
    :return: n_keys distribution feature dataframe.
    """
    key_df = segmented_player_df.select([*KEY_FEATURES, "segment_id"])

    features = (
        key_df
        .with_columns(
            pl.sum_horizontal(KEY_FEATURES).alias("n_keys"),
        )
        .group_by("segment_id", "n_keys")
        .agg(
            pl.count().alias("count")
        )
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over("segment_id")).alias("normalized_count")
        )
        .pivot(
            values="normalized_count",
            index="segment_id",
            columns="n_keys",
            aggregate_function="first"
        )
        .fill_null(0)
    )
    # add missing columns (e.g. if not all 6 buttons are pressed)
    n_values = [str(n) for n in range(0, len(KEY_FEATURES) + 1)]
    features = (
        features
        .with_columns([
            pl.lit(0).alias(col) for col in n_values if col not in features.columns
        ])
        .rename({str(i): f"keys_down_{i}" for i in range(0, len(KEY_FEATURES) + 1)})
        .fill_null(0)
        .sort("segment_id")
    )
    # sort columns
    features = (
        features
        .select(sorted(features.columns))
    )
    return features
