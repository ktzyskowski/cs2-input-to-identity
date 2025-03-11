from collections import Counter
from itertools import pairwise

import numpy as np
import polars as pl

from collection.parser.segment_parser.constants import KEY_FEATURES


def extract_entropy(segmented_player_df: pl.DataFrame) -> pl.Series:
    """Compute the Shannon entropy for bigram sequence of key presses.

    :param segmented_player_df:
    :return:
    """
    # extract N key presses per tick
    key_presses_df = extract_key_presses(segmented_player_df)

    # create bigrams
    bigrams_df = (
        key_presses_df.with_columns(
            pl.col("presses")
            .map_elements(lambda presses: (
                [f"{a}->{b}" for a, b in pairwise(presses)] if not presses.is_empty() else []
            ), return_dtype=pl.List(pl.Utf8))
            .alias("bigrams")
        )
    )

    bigram_frequency_distribution = compute_bigram_frequency_distribution(bigrams_df)
    entropy = compute_entropy(bigrams_df, bigram_frequency_distribution)
    return entropy


def extract_key_presses(segmented_player_df: pl.DataFrame) -> pl.DataFrame:
    key_presses = (
        segmented_player_df
        # detect rising edges
        .with_columns([
            ((pl.col(key).shift().over("segment_id").fill_null(0) == 0) & (pl.col(key) == 1))
            .alias(f"{key}_rising_edge")
            for key in KEY_FEATURES
        ])
        # select key presses when rising edge occurs
        .select(["segment_id"] + [
            pl.when(pl.col(f"{key}_rising_edge"))
                .then(pl.lit(key))
                .otherwise(pl.lit(None))
                .alias(f"{key}_event")
            for key in KEY_FEATURES
        ])
        # group by segment ID
        .group_by("segment_id")
        .agg([
            # concatenate rows together that have non-null key presses (i.e. rising edge)
            pl.concat_list([
                pl.col(f"{key}_event") for key in KEY_FEATURES
            ])
            .list.explode()  # flatten nested lists (list[list[str]] → list[str])
            .drop_nulls()
            .alias("presses")
        ])
        .sort("segment_id")
    )
    return key_presses


def compute_bigram_frequency_distribution(bigrams_df: pl.DataFrame) -> Counter:
    bigram_frequency_distribution = Counter(
        bigrams_df
        .select("bigrams")
        .to_series()
        .explode()
        .drop_nulls()
        .to_list()
    )
    return bigram_frequency_distribution


def compute_entropy(bigrams_df: pl.DataFrame, bigram_frequency_distribution: Counter) -> pl.Series:
    # compute shannon entropy
    # -sum(p(x)*log(p(x))
    n_bigrams = sum(bigram_frequency_distribution.values())
    entropy = (
        bigrams_df
        .select("bigrams")
        .to_series()
        .map_elements(
            lambda xs: -sum(
                bigram_frequency_distribution[x] / n_bigrams
                * np.log(bigram_frequency_distribution[x] / n_bigrams)
                for x in xs
            ),
            return_dtype=pl.Float32
        )
        .alias("entropy")
    )
    return entropy
