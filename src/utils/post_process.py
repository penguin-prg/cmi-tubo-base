import numpy as np
import polars as pl
from scipy.signal import find_peaks
import pandas as pd

def post_process_for_seg(
    keys: list[str], preds: np.ndarray, score_th: float = 0.01, distance: int = 5000, penguin_pp: bool = False
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    starts = np.array(list(map(lambda x: int(x.split("_")[2]), keys)))
    ends = np.array(list(map(lambda x: int(x.split("_")[3]), keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    dfs = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx] #.reshape(-1, 2)
        this_series_starts = starts[series_idx]
        this_series_ends = ends[series_idx]

        # 集約
        max_step = this_series_ends.max()
        agg_preds = np.zeros((max_step, 2))
        agg_counts = np.zeros((max_step, 2))
        for i, (start, end) in enumerate(zip(this_series_starts, this_series_ends)):
            agg_preds[start:end] += this_series_preds[i]
            agg_counts[start:end] += 1
        this_series_preds = agg_preds / agg_counts

        if penguin_pp:
            oof_df = pd.DataFrame({
                "series_id": series_id,
                "step": np.arange(len(this_series_preds)),
                "onset_oof": this_series_preds[:, 0],
                "wakeup_oof": this_series_preds[:, 1],
            })
            dfs.append(oof_df)

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    if penguin_pp:
        final_dfs = []
        for train in dfs:
            train[["wakeup_oof", "onset_oof"]] *= 10
            train["step"] = train["step"].astype(int)
            train = train[train["step"] % 12 == 6]

            this_dfs = []
            df = train[["series_id", "step", "wakeup_oof"]].copy()
            df["event"] = "wakeup"
            df["score"] = df["wakeup_oof"]
            df = df[df["score"]>0.005]
            if len(df) == 0:
                df = pd.DataFrame({"series_id": [0], "step": [0], "event": ["wakeup"], "score": [0]})
            this_dfs.append(df[['series_id', 'step', 'event', 'score']])

            df = train[["series_id", "step", "onset_oof"]].copy()
            df["event"] = "onset"
            df["score"] = df["onset_oof"]
            df = df[df["score"]>0.005]
            if len(df) == 0:
                df = pd.DataFrame({"series_id": [0], "step": [0], "event": ["onset"], "score": [0]})
            this_dfs.append(df[['series_id', 'step', 'event', 'score']])

            train = pd.concat(this_dfs).reset_index(drop=True)

            sub = dynamic_range_nms(train)
            sub["score"] = sub["reduced_score"]
            final_dfs.append(sub)

        sub = pd.concat(final_dfs).reset_index(drop=True)
        sub["row_id"] = sub.index
        sub = sub[["row_id", "series_id", "step", "event", "score"]]
        return sub, dfs

    else:
        return sub_df



from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt
import gc
from typing import Optional
from scipy.interpolate import interp1d


RANGE = 261
COEFF = 28
EXP = 5

def dynamic_range_nms(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamic-Range NMS

    Parameters
    ----------
    df : pd.DataFrame
        単一のseries_idに対する提出形式
    """
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    used = []
    used_scores = []
    reduce_rate = np.ones(df["step"].max() + 1000)
    for _ in range(min(len(df), 1000)):
        df["reduced_score"] = df["score"] / reduce_rate[df["step"]]
        best_score = df["reduced_score"].max()
        best_idx = df["reduced_score"].idxmax()
        best_step = df.loc[best_idx, "step"]
        used.append(best_idx)
        used_scores.append(best_score)

        for r in range(1, int(RANGE)):
            reduce = ((RANGE - r) / RANGE) ** EXP * COEFF
            reduce_rate[best_step + r] += reduce
            if best_step - r >= 0:
                reduce_rate[best_step - r] += reduce
        reduce_rate[best_step] = 1e10
    df = df.iloc[used].copy()
    df["reduced_score"] = used_scores
    return df
