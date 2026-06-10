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


@dataclass
class VPINResult:
    """VPIN 计算结果"""

    vpin_df: pl.DataFrame
    trades_with_sign: pl.DataFrame
    volume_buckets: pl.DataFrame
    bucket_size: int
    n_buckets: int
    avg_vpin: float
    max_vpin: float

    def __repr__(self) -> str:
        return (
            f"VPINResult(buckets={self.n_buckets}, bucket_size={self.bucket_size}, "
            f"avg_vpin={self.avg_vpin:.4f}, max_vpin={self.max_vpin:.4f})"
        )


class MicrostructureAnalytics:
    """
    微观结构分析引擎

    基于 Polars 对 ITCH 消息流和订单簿快照进行聚合分析，
    计算各类高频交易微观结构指标。
    """

    def __init__(self):
        self._snapshots_df: Optional[pl.DataFrame] = None
        self._messages_df: Optional[pl.DataFrame] = None
        self._trades_df: Optional[pl.DataFrame] = None

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

        if "trades" in parsed_data:
            self._trades_df = parsed_data["trades"]

    def load_trades(self, trades_df: pl.DataFrame) -> None:
        """
        直接加载逐笔成交数据

        Parameters
        ----------
        trades_df : pl.DataFrame
            包含 timestamp, price_float, shares, side 字段的逐笔成交表
        """
        self._trades_df = trades_df

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

    def lee_ready_tick_test(
        self,
        trades_df: Optional[pl.DataFrame] = None,
        snapshots_df: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Lee-Ready Tick Test 算法：判定每笔成交的发起方

        算法规则：
          1. 成交价 > 上一笔成交价 (uptick)   → 买方发起 (+1)
          2. 成交价 < 上一笔成交价 (downtick) → 卖方发起 (-1)
          3. 成交价 == 上一笔成交价 (zero tick)：
             a. 若离卖一价更近 → 买方发起 (+1)
             b. 若离买一价更近 → 卖方发起 (-1)
             c. 正好等于中间价 → 沿用上一笔方向

        Parameters
        ----------
        trades_df : Optional[pl.DataFrame]
            逐笔成交数据，需包含 timestamp, price_float, shares
        snapshots_df : Optional[pl.DataFrame]
            订单簿快照数据，用于获取 midprice/best_bid/best_ask

        Returns
        -------
        pl.DataFrame
            带 trade_sign (+1/-1) 列的成交数据
        """
        if trades_df is None:
            trades_df = self._trades_df
        if trades_df is None or trades_df.is_empty():
            return pl.DataFrame()

        df = trades_df.clone().sort("timestamp")

        df = df.with_columns(
            pl.col("price_float").diff().alias("price_diff")
        )

        if snapshots_df is None:
            snapshots_df = self._snapshots_df

        midprice_series = None
        if snapshots_df is not None and not snapshots_df.is_empty():
            snap = snapshots_df.sort("timestamp").select(
                ["timestamp", "midprice", "best_bid", "best_ask"]
            )
            df = df.join_asof(snap, on="timestamp", strategy="backward")
            midprice_series = pl.col("midprice")
        else:
            df = df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("midprice"),
                pl.lit(None).cast(pl.Float64).alias("best_bid"),
                pl.lit(None).cast(pl.Float64).alias("best_ask"),
            )

        df = df.with_columns(
            pl.when(pl.col("price_diff") > 0)
            .then(pl.lit(1))
            .when(pl.col("price_diff") < 0)
            .then(pl.lit(-1))
            .when(
                (pl.col("price_diff") == 0)
                & (pl.col("midprice").is_not_null())
                & (pl.col("best_bid").is_not_null())
                & (pl.col("best_ask").is_not_null())
            )
            .then(
                pl.when(
                    (pl.col("price_float") - pl.col("best_ask")).abs()
                    < (pl.col("price_float") - pl.col("best_bid")).abs()
                )
                .then(pl.lit(1))
                .when(
                    (pl.col("price_float") - pl.col("best_ask")).abs()
                    > (pl.col("price_float") - pl.col("best_bid")).abs()
                )
                .then(pl.lit(-1))
                .otherwise(pl.lit(None))
            )
            .otherwise(pl.lit(None))
            .alias("trade_sign_raw")
        )

        df = df.with_columns(
            pl.col("trade_sign_raw")
            .forward_fill()
            .fill_null(1)
            .alias("trade_sign")
        )

        df = df.with_columns(
            (pl.col("trade_sign") * pl.col("shares")).alias("signed_volume")
        )

        return df

    def compute_volume_buckets(
        self,
        trades_with_sign: pl.DataFrame,
        bucket_size: Optional[int] = None,
        n_buckets: int = 50,
    ) -> pl.DataFrame:
        """
        按等量成交量（Volume Buckets）聚合，而非物理时间切片

        Parameters
        ----------
        trades_with_sign : pl.DataFrame
            经过 Lee-Ready 判定、带 trade_sign 列的成交数据
        bucket_size : Optional[int]
            每个桶的目标成交量。若为 None 则自动根据 n_buckets 计算
        n_buckets : int
            期望的桶数量（当 bucket_size 为 None 时使用）

        Returns
        -------
        pl.DataFrame
            每个成交量桶的聚合结果：start_ts, end_ts, V_buy, V_sell, |V_buy-V_sell|
        """
        if trades_with_sign.is_empty():
            return pl.DataFrame()

        df = trades_with_sign.sort("timestamp")

        if bucket_size is None:
            total_volume = df["shares"].sum()
            bucket_size = max(1, int(total_volume / max(1, n_buckets)))

        df = df.with_columns(
            pl.col("shares").cum_sum().alias("cum_vol")
        )

        df = df.with_columns(
            (pl.col("cum_vol") // int(bucket_size)).alias("bucket_idx")
        )

        df = df.with_columns(
            pl.when(pl.col("trade_sign") > 0)
            .then(pl.col("shares"))
            .otherwise(pl.lit(0))
            .alias("buy_volume"),
            pl.when(pl.col("trade_sign") < 0)
            .then(pl.col("shares"))
            .otherwise(pl.lit(0))
            .alias("sell_volume"),
        )

        buckets = df.group_by("bucket_idx", maintain_order=True).agg(
            pl.col("timestamp").min().alias("start_ts"),
            pl.col("timestamp").max().alias("end_ts"),
            pl.col("buy_volume").sum().alias("buy_volume"),
            pl.col("sell_volume").sum().alias("sell_volume"),
            pl.count().alias("n_trades"),
        )

        buckets = buckets.with_columns(
            pl.col("buy_volume").cast(pl.Int64).alias("buy_volume_i64"),
            pl.col("sell_volume").cast(pl.Int64).alias("sell_volume_i64"),
        )

        buckets = buckets.with_columns(
            (pl.col("buy_volume_i64") - pl.col("sell_volume_i64")).alias("net_volume"),
            (pl.col("buy_volume_i64") - pl.col("sell_volume_i64")).abs().alias("abs_imbalance"),
            pl.col("buy_volume").cast(pl.Int64).cum_sum().alias("cum_buy"),
            pl.col("sell_volume").cast(pl.Int64).cum_sum().alias("cum_sell"),
        )

        buckets = buckets.drop(["buy_volume_i64", "sell_volume_i64"])

        buckets = buckets.with_columns(
            pl.col("start_ts").cast(pl.Datetime("us")).alias("start_datetime"),
            pl.col("end_ts").cast(pl.Datetime("us")).alias("end_datetime"),
        )

        return buckets

    def compute_vpin(
        self,
        bucket_size: Optional[int] = None,
        n_buckets: int = 50,
        vpin_window: int = 50,
    ) -> VPINResult:
        """
        计算 VPIN（知情交易同步概率，Volume-Synchronized Probability of Informed Trading）

        VPIN 经典公式 (Easley et al. 2011)：
            VPIN = (1 / n) * Σ (|V_buy,i - V_sell,i|) / V_bucket

        其中 n = 滚动窗口中的桶数量

        Parameters
        ----------
        bucket_size : Optional[int]
            每桶目标成交量。为 None 时自动按总成交量 / n_buckets 计算
        n_buckets : int
            期望的桶数量（自动模式用）
        vpin_window : int
            VPIN 滚动窗口大小（桶数量），默认 50

        Returns
        -------
        VPINResult
            包含 vpin_df, trades_with_sign, volume_buckets 等完整结果
        """
        if self._trades_df is None or self._trades_df.is_empty():
            return VPINResult(
                vpin_df=pl.DataFrame(),
                trades_with_sign=pl.DataFrame(),
                volume_buckets=pl.DataFrame(),
                bucket_size=0,
                n_buckets=0,
                avg_vpin=0.0,
                max_vpin=0.0,
            )

        trades_with_sign = self.lee_ready_tick_test()

        buckets = self.compute_volume_buckets(
            trades_with_sign, bucket_size=bucket_size, n_buckets=n_buckets
        )

        if buckets.is_empty():
            return VPINResult(
                vpin_df=pl.DataFrame(),
                trades_with_sign=trades_with_sign,
                volume_buckets=pl.DataFrame(),
                bucket_size=0,
                n_buckets=0,
                avg_vpin=0.0,
                max_vpin=0.0,
            )

        buckets = buckets.with_columns(
            (pl.col("buy_volume") + pl.col("sell_volume")).alias("total_volume"),
        )

        buckets = buckets.with_columns(
            pl.when(pl.col("total_volume") > 0)
            .then(pl.col("abs_imbalance") / pl.col("total_volume"))
            .otherwise(0.0)
            .alias("imbalance_ratio")
        )

        vpin_df = buckets.with_columns(
            pl.col("imbalance_ratio")
            .rolling_mean(window_size=vpin_window, min_samples=1)
            .alias("vpin")
        )

        valid_vpin = vpin_df["vpin"].drop_nans()
        avg_vpin = float(valid_vpin.mean()) if len(valid_vpin) > 0 else 0.0
        max_vpin = float(valid_vpin.max()) if len(valid_vpin) > 0 else 0.0

        actual_bucket_size = int(buckets["total_volume"].mean() or 0)

        return VPINResult(
            vpin_df=vpin_df,
            trades_with_sign=trades_with_sign,
            volume_buckets=buckets,
            bucket_size=actual_bucket_size,
            n_buckets=len(buckets),
            avg_vpin=avg_vpin,
            max_vpin=max_vpin,
        )

    def compute_candlesticks_from_snapshots(
        self,
        rule: str = "1min",
    ) -> pl.DataFrame:
        """
        从订单簿快照聚合生成 K 线数据

        Parameters
        ----------
        rule : str
            K 线时间周期，如 "1s", "1min", "5min"

        Returns
        -------
        pl.DataFrame
            包含 datetime, open, high, low, close, volume 的 K 线数据
        """
        if self._snapshots_df is None or self._snapshots_df.is_empty():
            return pl.DataFrame()

        df = self._snapshots_df.clone().sort("datetime")

        ohlc = df.group_by_dynamic("datetime", every=rule).agg(
            pl.col("midprice").first().alias("open"),
            pl.col("midprice").max().alias("high"),
            pl.col("midprice").min().alias("low"),
            pl.col("midprice").last().alias("close"),
        )

        if self._trades_df is not None and not self._trades_df.is_empty():
            trades = self._trades_df.clone().with_columns(
                pl.col("timestamp").cast(pl.Datetime("us")).alias("datetime")
            ).sort("datetime")

            vol = trades.group_by_dynamic("datetime", every=rule).agg(
                pl.col("shares").sum().alias("volume")
            )

            ohlc = ohlc.join_asof(vol, on="datetime", strategy="backward")
        else:
            ohlc = ohlc.with_columns(pl.lit(0).alias("volume"))

        return ohlc
