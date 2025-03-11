import polars as pl

from collection.parser.segment_parser.util import fill_segment_ids


def extract_count(mouse_df: pl.DataFrame) -> pl.DataFrame:
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
        # group by segment ID, count up number of unique mouse_movement IDs
        .group_by("segment_id")
        .agg(
            pl.col("mouse_movement_id")
            .n_unique()
            .alias("n_mouse_movements")
        )
    )
    return fill_segment_ids(mouse_df, features)
