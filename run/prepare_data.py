import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

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
    "large_diff_count_1",
    "large_diff_count_5",
    "large_diff_max_1",
    "large_diff_max_5",
    "anglez_range_1",
    "anglez_range_5",
    "enmo_mean_1",
    "enmo_mean_5",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


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
        for m in [1, 5]:
            series_lf = series_lf.with_columns([        
                pl.col("large_diff").rolling_mean(window_size=12*m, min_periods=1, center=True).over("series_id").fill_null(0).alias(f"large_diff_count_{m}")
            ])
            series_lf = series_lf.with_columns([        
                pl.col("large_diff").rolling_max(window_size=12*m, min_periods=1, center=True).over("series_id").fill_null(0).alias(f"large_diff_max_{m}")
            ])

            series_lf = series_lf.with_columns([
                (pl.col("anglez").rolling_max(window_size=12*m, min_periods=1, center=True).over("series_id").fill_null(0) \
                - pl.col("anglez").rolling_min(window_size=12*m, min_periods=1, center=True).over("series_id").fill_null(0)) \
                .alias(f"anglez_range_{m}") / ANGLEZ_STD
            ])

            series_lf = series_lf.with_columns([
                pl.col("enmo").rolling_mean(window_size=12*m, min_periods=1, center=True).over("series_id").fill_null(0).alias(f"enmo_mean_{m}") / ENMO_STD
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
                pl.col("large_diff_count_1"),
                pl.col("large_diff_count_5"),
                pl.col("large_diff_max_1"),
                pl.col("large_diff_max_5"),
                pl.col("anglez_range_1"),
                pl.col("anglez_range_5"),
                pl.col("enmo_mean_1"),
                pl.col("enmo_mean_5"),
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


if __name__ == "__main__":
    main()
