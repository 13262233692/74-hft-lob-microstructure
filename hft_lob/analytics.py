"""
微观结构分析模块
基于 Polars 的高性能数据管道，计算订单簿深度、价差、订单流等微观结构指标
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import polars as pl


@dataclass
class MicrostructureMetrics:
    """微观结构指标集合"""

    avg_spread: float
    median_spread: float
    spread_std: float
    avg_midprice: float
    midprice_volatility: float
    avg_bid_depth_1: float
    avg_ask_depth_1: float
    total_depth_bid: float
    total_depth_ask: float
    order_imbalance: float
    n_snapshots: int


class MicrostructureAnalytics:
    """
    微观结构分析引擎

    基于 Polars 对 ITCH 消息流和订单簿快照进行聚合分析，
    计算各类高频交易微观结构指标。
    """

    def __init__(self):
        self._snapshots_df: Optional[pl.DataFrame] = None
        self._messages_df: Optional[pl.DataFrame] = None

    def load_snapshots(self, snapshots_df: pl.DataFrame) -> None:
        """
        加载订单簿快照数据

        Parameters
        ----------
        snapshots_df : pl.DataFrame
            L3OrderBook.to_polars() 返回的快照 DataFrame
        """
        self._snapshots_df = snapshots_df

    def load_parsed_messages(self, parsed_data: dict) -> None:
        """
        加载解析后的 ITCH 消息

        Parameters
        ----------
        parsed_data : dict
            ITCHParser.to_polars() 返回的 DataFrame 字典
        """
        common_cols = ["timestamp", "datetime", "msg_type_name"]
        dfs = []
        for name, df in parsed_data.items():
            df = df.with_columns(pl.lit(name).alias("msg_type_name"))
            keep = [c for c in common_cols if c in df.columns]
            dfs.append(df.select(keep))
        if dfs:
            self._messages_df = pl.concat(dfs, how="diagonal_relaxed")

    def compute_metrics(self) -> MicrostructureMetrics:
        """
        计算核心微观结构指标

        Returns
        -------
        MicrostructureMetrics
            包含价差、深度、订单不平衡等指标
        """
        if self._snapshots_df is None or self._snapshots_df.is_empty():
            return MicrostructureMetrics(
                avg_spread=0.0,
                median_spread=0.0,
                spread_std=0.0,
                avg_midprice=0.0,
                midprice_volatility=0.0,
                avg_bid_depth_1=0.0,
                avg_ask_depth_1=0.0,
                total_depth_bid=0.0,
                total_depth_ask=0.0,
                order_imbalance=0.0,
                n_snapshots=0,
            )

        df = self._snapshots_df

        avg_spread = df["spread"].mean() or 0.0
        median_spread = df["spread"].median() or 0.0
        spread_std = df["spread"].std() or 0.0

        avg_midprice = df["midprice"].mean() or 0.0
        mid_returns = df["midprice"].pct_change().drop_nans()
        midprice_volatility = float(mid_returns.std()) * np.sqrt(86400) if len(mid_returns) > 1 else 0.0

        avg_bid_depth_1 = df["bid_size_1"].drop_nulls().mean() or 0.0
        avg_ask_depth_1 = df["ask_size_1"].drop_nulls().mean() or 0.0

        bid_size_cols = [f"bid_size_{i}" for i in range(1, 11)]
        ask_size_cols = [f"ask_size_{i}" for i in range(1, 11)]

        total_bid = df.select(pl.sum_horizontal(bid_size_cols)).to_series().drop_nulls()
        total_ask = df.select(pl.sum_horizontal(ask_size_cols)).to_series().drop_nulls()

        total_depth_bid = float(total_bid.mean()) if len(total_bid) > 0 else 0.0
        total_depth_ask = float(total_ask.mean()) if len(total_ask) > 0 else 0.0

        if total_depth_bid + total_depth_ask > 0:
            order_imbalance = (total_depth_bid - total_depth_ask) / (total_depth_bid + total_depth_ask)
        else:
            order_imbalance = 0.0

        return MicrostructureMetrics(
            avg_spread=float(avg_spread),
            median_spread=float(median_spread),
            spread_std=float(spread_std),
            avg_midprice=float(avg_midprice),
            midprice_volatility=float(midprice_volatility),
            avg_bid_depth_1=float(avg_bid_depth_1),
            avg_ask_depth_1=float(avg_ask_depth_1),
            total_depth_bid=float(total_depth_bid),
            total_depth_ask=float(total_depth_ask),
            order_imbalance=float(order_imbalance),
            n_snapshots=len(df),
        )

    def resample_snapshots(
        self, rule: str = "1s"
    ) -> pl.DataFrame:
        """
        按时间粒度重采样订单簿快照

        Parameters
        ----------
        rule : str
            重采样频率，如 "100ms", "1s", "1m"

        Returns
        -------
        pl.DataFrame
            重采样后的快照数据
        """
        if self._snapshots_df is None:
            return pl.DataFrame()

        df = self._snapshots_df

        bid_size_cols = [f"bid_size_{i}" for i in range(1, 11)]
        ask_size_cols = [f"ask_size_{i}" for i in range(1, 11)]
        bid_price_cols = [f"bid_price_{i}" for i in range(1, 11)]
        ask_price_cols = [f"ask_price_{i}" for i in range(1, 11)]

        agg_exprs = [
            pl.col("spread").mean().alias("spread_mean"),
            pl.col("spread").last().alias("spread_last"),
            pl.col("midprice").last().alias("midprice_last"),
            pl.col("best_bid").last().alias("best_bid"),
            pl.col("best_ask").last().alias("best_ask"),
        ]

        for col in bid_size_cols + ask_size_cols:
            agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))

        for col in bid_price_cols + ask_price_cols:
            agg_exprs.append(pl.col(col).last().alias(f"{col}_last"))

        return (
            df.sort("datetime")
            .group_by_dynamic("datetime", every=rule)
            .agg(agg_exprs)
        )

    def compute_order_flow_imbalance(
        self, window_us: int = 10_000_000
    ) -> pl.DataFrame:
        """
        计算订单流不平衡指标 (OFI)

        Parameters
        ----------
        window_us : int
            滚动窗口大小（微秒），默认 10 秒

        Returns
        -------
        pl.DataFrame
            包含 OFI 指标的时间序列
        """
        if self._snapshots_df is None or self._snapshots_df.is_empty():
            return pl.DataFrame()

        df = self._snapshots_df.sort("timestamp").with_columns(
            pl.col("bid_size_1").fill_null(0),
            pl.col("ask_size_1").fill_null(0),
            pl.col("bid_price_1").fill_null(0),
            pl.col("ask_price_1").fill_null(0),
        )

        df = df.with_columns(
            pl.col("bid_price_1").diff().alias("dbp"),
            pl.col("bid_size_1").diff().alias("dbs"),
            pl.col("ask_price_1").diff().alias("dap"),
            pl.col("ask_size_1").diff().alias("das"),
        )

        def ofi_buy(dbp: pl.Expr, dbs: pl.Expr) -> pl.Expr:
            return (
                pl.when(dbp > 0).then(pl.col("bid_size_1"))
                .when(dbp == 0).then(dbs)
                .otherwise(pl.lit(0))
            )

        def ofi_sell(dap: pl.Expr, das: pl.Expr) -> pl.Expr:
            return (
                pl.when(dap < 0).then(pl.col("ask_size_1"))
                .when(dap == 0).then(das)
                .otherwise(pl.lit(0))
            )

        df = df.with_columns(
            ofi_buy(pl.col("dbp"), pl.col("dbs")).alias("ofi_buy"),
            ofi_sell(pl.col("dap"), pl.col("das")).alias("ofi_sell"),
        )

        df = df.with_columns(
            (pl.col("ofi_buy") - pl.col("ofi_sell")).alias("ofi")
        )

        window_ns = window_us * 1000

        return (
            df.sort("datetime")
            .group_by_dynamic("datetime", every=f"{window_us // 1_000_000}s")
            .agg(
                pl.col("ofi").sum().alias("ofi_sum"),
                pl.col("ofi_buy").sum().alias("ofi_buy_sum"),
                pl.col("ofi_sell").sum().alias("ofi_sell_sum"),
                pl.col("midprice").last().alias("midprice"),
            )
        )

    def compute_depth_profile(self) -> pl.DataFrame:
        """
        计算平均累计深度曲线

        Returns
        -------
        pl.DataFrame
            各价位档位的累计平均深度
        """
        if self._snapshots_df is None or self._snapshots_df.is_empty():
            return pl.DataFrame()

        df = self._snapshots_df
        rows = []

        for i in range(1, 11):
            bid_col = f"bid_size_{i}"
            ask_col = f"ask_size_{i}"
            bid_p_col = f"bid_price_{i}"
            ask_p_col = f"ask_price_{i}"

            avg_bid_size = df[bid_col].drop_nulls().mean() or 0.0
            avg_ask_size = df[ask_col].drop_nulls().mean() or 0.0
            avg_bid_price = df[bid_p_col].drop_nulls().mean() or 0.0
            avg_ask_price = df[ask_p_col].drop_nulls().mean() or 0.0

            rows.append(
                {
                    "level": i,
                    "side": "bid",
                    "avg_price": float(avg_bid_price),
                    "avg_size": float(avg_bid_size),
                }
            )
            rows.append(
                {
                    "level": i,
                    "side": "ask",
                    "avg_price": float(avg_ask_price),
                    "avg_size": float(avg_ask_size),
                }
            )

        return pl.DataFrame(rows)

    def message_statistics(self) -> pl.DataFrame:
        """
        统计各类 ITCH 消息的数量和时间分布

        Returns
        -------
        pl.DataFrame
            消息类型统计
        """
        if self._messages_df is None:
            return pl.DataFrame()

        return (
            self._messages_df.group_by("msg_type_name")
            .agg(
                pl.count().alias("count"),
                pl.col("timestamp").min().alias("first_ts"),
                pl.col("timestamp").max().alias("last_ts"),
            )
            .sort("count", descending=True)
        )
