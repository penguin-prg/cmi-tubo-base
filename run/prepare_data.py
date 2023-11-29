import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm
import gc

from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
    "same_count",
    "large_diff_count",
    "large_diff_max",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


import polars as pl


def transform(df, night_offset=20):
    return (
        df.with_columns(
            [
                pl.col("timestamp").str.slice(-5, 3).cast(pl.Int8).alias("tz_offset"),
            ]
        )
        .with_columns(
            [
                (pl.col("tz_offset") == -4).alias("is_dst"),
            ]
        )
        .with_columns(
            [
                pl.col("timestamp")
                .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z")
                .alias("timestamp"),
            ]
        )
        .with_columns(
            [
                (pl.col("timestamp").dt.year() - 2000).cast(pl.Int8).alias("year"),
                pl.col("timestamp").dt.month().cast(pl.Int8).alias("month"),
                pl.col("timestamp").dt.day().cast(pl.Int8).alias("day"),
                pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hour"),
                pl.col("timestamp").dt.minute().cast(pl.Int8).alias("minute"),
                pl.col("timestamp").dt.second().cast(pl.Int8).alias("second"),
                pl.col("timestamp").dt.weekday().cast(pl.Int8).alias("weekday"),
            ]
        )
        .with_columns(  # 正午をまたいで日付を調整
            pl.when(pl.col("hour") < night_offset)
            .then(pl.col("timestamp"))
            .otherwise(pl.col("timestamp") + pl.duration(days=1))
            .dt.date()
            .alias("night_group"),
        )
        .with_columns(
            [
                (
                    pl.col("series_id")
                    + pl.lit("_")
                    + pl.col("night_group").cast(pl.Datetime).dt.strftime("%Y%m%d")
                ).alias("group_id"),
            ]
        )
        .with_columns(
            [
                pl.col("timestamp").cumcount().over("group_id").alias("norm_step"),
            ]
        )
        .drop(["night_group"])
    )


def transform_series(df):
    return transform(df).with_columns(
        [
            (pl.col("enmo") == 0).alias("is_enmo_clipped"),
        ]
    )


def transform_events(df):
    return (
        transform(df)
        .with_columns(
            [
                pl.col("night").cast(pl.UInt32).alias("night"),
            ]
        )
        .pivot(["step", "timestamp", "tz_offset"], ["series_id", "group_id", "night"], "event")
    )

def add_heuristic_feature(
    df,
    group_col="series_id",
    day_group_col="group_id",
    term1=(5 * 60) // 5,
    term2=(30 * 60) // 5,
    term3=(60 * 60) // 5,
    min_threshold=0.005,
    max_threshold=0.04,
    center=True,
):
    return (
        df.with_columns(
            [
                pl.col("anglez").diff(1).abs().over(group_col).alias("anglez_diff"),
                pl.col("enmo").diff(1).abs().over(group_col).alias("enmo_diff"),
            ]
        )
        .with_columns(
            [
                pl.col("anglez_diff")
                .rolling_median(term1, center=center)  # 5 min window
                .over(group_col)
                .alias("anglez_diff_median_5min"),
                pl.col("enmo_diff")
                .rolling_median(term1, center=center)  # 5 min window
                .over(group_col)
                .alias("enmo_diff_median_5min"),
            ]
        )
        .with_columns(
            [
                pl.col("anglez_diff_median_5min")
                .quantile(0.1)
                .clip(min_threshold, max_threshold)
                .over(day_group_col)
                .alias("critical_threshold")
            ]
        )
        .with_columns(
            [
                (pl.col("anglez_diff_median_5min") < pl.col("critical_threshold") * 15)
                .over(group_col)
                .alias("is_static")
            ]
        )
        .with_columns(
            [
                pl.col("is_static")
                .cast(pl.Int32)
                .rolling_sum(term2, center=center)
                .over(group_col)
                .alias("is_static_sum_30min"),
            ]
        )
        .with_columns(
            [(pl.col("is_static_sum_30min") == ((30 * 60) // 5)).over(group_col).alias("tmp")]
        )
        .with_columns(
            [
                pl.col("tmp").shift(term2 // 2).over(group_col).alias("tmp_left"),
                pl.col("tmp").shift(-(term2 // 2)).over(group_col).alias("tmp_right"),
            ]
        )
        .with_columns(
            [
                (pl.col("tmp_left") | pl.col("tmp_right")).alias("is_sleep_block"),
            ]
        )
        .drop(["tmp", "tmp_left", "tmp_right"])
        .with_columns([pl.col("is_sleep_block").not_().alias("is_gap")])
        .with_columns(
            [
                pl.col("is_gap")
                .cast(pl.Int32)
                .rolling_sum(term3, center=center)
                .over(group_col)
                .alias("gap_length")
            ]
        )
        .with_columns([(pl.col("gap_length") == term3).over(group_col).alias("tmp")])
        .with_columns(
            [
                pl.col("tmp").shift(term3 // 2).over(group_col).alias("tmp_left"),
                pl.col("tmp").shift(-(term3 // 2)).over(group_col).alias("tmp_right"),
            ]
        )
        .with_columns(
            [
                (pl.col("tmp_left") | pl.col("tmp_right")).alias("is_large_gap"),
            ]
        )
        .drop(["tmp", "tmp_left", "tmp_right"])
        .with_columns([pl.col("is_large_gap").not_().alias("is_sleep_episode")])
        #
        # extract longest sleep episode
        #
        .with_columns(
            [
                # extract false->true transition
                (
                    (
                        pl.col("is_sleep_episode")
                        & pl.col("is_sleep_episode")
                        .shift_and_fill(pl.lit(False), periods=1)
                        .not_()
                    )
                    .cumsum()
                    .over("group_id")
                ).alias("sleep_episode_id")
            ]
        )
        .with_columns(
            [
                pl.col("is_sleep_episode")
                .sum()
                .over(["group_id", "sleep_episode_id"])
                .alias("sleep_episode_length")
            ]
        )
        .with_columns(
            [
                pl.col("sleep_episode_length")
                .max()
                .over(["group_id"])
                .alias("max_sleep_episode_length")
            ]
        )
        .with_columns(
            [
                (
                    pl.col("is_sleep_episode")
                    & (pl.col("sleep_episode_length") == pl.col("max_sleep_episode_length"))
                ).alias("is_longest_sleep_episode")
            ]
        )
    )


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]

def calculate_same_count(df):
    DAY_STEPS = 12 * 60 * 24
    n_days = int(len(df) // DAY_STEPS) + 1
    # same_countカラムを初期化
    df = df.with_columns(pl.lit(0).alias("same_count"))

    for day in range(-n_days, n_days + 1):
        if day == 0:
            continue
        # anglezのDAY_STEPS * dayだけずらした差分を計算
        anglez_diff = df["anglez"].shift(DAY_STEPS * day).fill_null(1) - df["anglez"]
        # same_countをインクリメント
        df = df.with_columns(pl.when(anglez_diff == 0).then(1).otherwise(0).alias("_increment"))
        df = df.with_columns([
            (pl.col("same_count") + pl.col("_increment")).alias("same_count")
        ])

    # same_countをクリップして正規化
    df = df.with_columns(
        ((pl.col("same_count").clip(0, 5) - 2.5) / 2.5).alias("same_count")
    )

    # 中間カラムをドロップ
    return df.drop("_increment")


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = calculate_same_count(series_df)
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
        *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
    ).select("series_id", *FEATURE_NAMES)
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # large diff count
        series_lf = series_lf.with_columns([
            pl.col("anglez").diff().over("series_id").fill_null(0).abs().alias("anglez_diffabs"),
        ])
        series_lf = series_lf.with_columns([
            (pl.col("anglez_diffabs") > 5).cast(pl.Int32).alias("large_diff"),
        ])
        series_lf = series_lf.with_columns([        
            pl.col("large_diff").rolling_mean(window_size=10, min_periods=1).over("series_id").fill_null(0).alias("large_diff_count")
        ])
        series_lf = series_lf.with_columns(
            ((pl.col("large_diff_count") - 0.5) * 2).alias("large_diff_count")
        )
        series_lf = series_lf.with_columns([        
            pl.col("large_diff").rolling_max(window_size=12*5, min_periods=1).over("series_id").fill_null(0).alias("large_diff_max")
        ])

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )            
            .select([
                pl.col("series_id"), 
                pl.col("anglez"), 
                pl.col("enmo"), 
                pl.col("timestamp"), 
                pl.col("step"), 
                pl.col("large_diff_count"),
                pl.col("large_diff_max"),
            ])
            .collect(streaming=True)
            .sort(by=["series_id", "step"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df)

            # 特徴量をそれぞれnpyで保存
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)
    del series_df
    gc.collect()

    # heuristic features
    tr_series = pl.read_parquet(
        Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
    )
    tr_series = transform_series(tr_series)
    tr_series = add_heuristic_feature(tr_series)
    use_columns = [
        "series_id", "step", 
        "is_longest_sleep_episode", "is_sleep_block",]
    tr_series = tr_series[use_columns].fill_null(False)
    tr_series = tr_series.with_columns([
        (pl.col("is_sleep_block").cast(pl.Int32) * 2 - 1).alias("is_sleep_block"),
        (pl.col("is_longest_sleep_episode").cast(pl.Int32) * 2 - 1).alias("is_longest_sleep_episode"),
    ]).sort(by=["series_id", "step"])
    for series_id, this_series_df in tqdm(tr_series.group_by("series_id"), total=n_unique):
        series_dir = processed_dir / series_id
        save_each_series(this_series_df, use_columns[2:], series_dir)



if __name__ == "__main__":
    main()
