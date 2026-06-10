"""
Plotly 可视化模块
为量化研究员提供交互式的微观结构图表：订单簿深度图、价差时序图、中间价轨迹等
"""

from typing import Optional, Tuple

import numpy as np
import polars as pl

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None


RED_COLOR = "#EF4444"
GREEN_COLOR = "#22C55E"
BLUE_COLOR = "#3B82F6"
DARK_BG = "#0F172A"
GRID_COLOR = "#1E293B"
TEXT_COLOR = "#E2E8F0"


def _check_plotly():
    if go is None:
        raise ImportError("plotly 未安装，请先安装: pip install plotly")


def _apply_dark_theme(fig: "go.Figure") -> "go.Figure":
    """应用深色量化研报风格主题"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(family="Consolas, Monospace", color=TEXT_COLOR, size=11),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode="x unified",
    )
    return fig


class MicrostructureViz:
    """
    微观结构可视化引擎

    提供订单簿深度图、价差时序图、中间价轨迹、深度剖面图等
    多种交互式图表，支持导出为 HTML 或静态图片。
    """

    def __init__(self, snapshots_df: Optional[pl.DataFrame] = None):
        """
        Parameters
        ----------
        snapshots_df : Optional[pl.DataFrame]
            L3OrderBook.to_polars() 输出的快照数据
        """
        _check_plotly()
        self.snapshots_df = snapshots_df

    def set_snapshots(self, snapshots_df: pl.DataFrame) -> None:
        self.snapshots_df = snapshots_df

    def plot_spread_series(
        self,
        title: str = "Bid-Ask Spread 时间序列",
        resample_rule: Optional[str] = None,
    ) -> "go.Figure":
        """
        绘制买卖价差 (Spread) 时间序列图

        Parameters
        ----------
        title : str
            图表标题
        resample_rule : Optional[str]
            重采样频率，如 "1s", "100ms"

        Returns
        -------
        go.Figure
            Plotly 图表对象
        """
        if self.snapshots_df is None:
            raise ValueError("请先通过 set_snapshots() 设置快照数据")

        df = self.snapshots_df.clone()

        if resample_rule:
            df = (
                df.sort("datetime")
                .group_by_dynamic("datetime", every=resample_rule)
                .agg(
                    pl.col("spread").mean().alias("spread"),
                    pl.col("midprice").last().alias("midprice"),
                )
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["datetime"].to_list(),
                y=df["spread"].to_list(),
                mode="lines",
                name="Spread",
                line=dict(color=BLUE_COLOR, width=1),
                fill="tozeroy",
                fillcolor=f"rgba(59, 130, 246, 0.15)",
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="时间",
            yaxis_title="价差 (Spread)",
            height=400,
        )

        return _apply_dark_theme(fig)

    def plot_midprice_series(
        self,
        title: str = "中间价 (Midprice) 轨迹",
    ) -> "go.Figure":
        """
        绘制中间价时间序列

        Parameters
        ----------
        title : str
            图表标题

        Returns
        -------
        go.Figure
        """
        if self.snapshots_df is None:
            raise ValueError("请先设置快照数据")

        df = self.snapshots_df

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["datetime"].to_list(),
                y=df["midprice"].to_list(),
                mode="lines",
                name="Midprice",
                line=dict(color=GREEN_COLOR, width=1),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["datetime"].to_list(),
                y=df["best_bid"].to_list(),
                mode="lines",
                name="Best Bid",
                line=dict(color=RED_COLOR, width=0.8, dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["datetime"].to_list(),
                y=df["best_ask"].to_list(),
                mode="lines",
                name="Best Ask",
                line=dict(color=GREEN_COLOR, width=0.8, dash="dash"),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="时间",
            yaxis_title="价格",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return _apply_dark_theme(fig)

    def plot_order_book_depth(
        self,
        snapshot_idx: int = -1,
        title: str = "L3 订单簿十档深度",
    ) -> "go.Figure":
        """
        绘制某一时刻的十档订单簿深度图（阶梯状累积深度）

        Parameters
        ----------
        snapshot_idx : int
            快照索引，-1 表示最后一个
        title : str
            图表标题

        Returns
        -------
        go.Figure
        """
        if self.snapshots_df is None:
            raise ValueError("请先设置快照数据")

        df = self.snapshots_df
        if snapshot_idx < 0:
            snapshot_idx = len(df) + snapshot_idx

        row = df.row(snapshot_idx, named=True)

        bid_prices = []
        bid_sizes = []
        ask_prices = []
        ask_sizes = []

        for i in range(1, 11):
            bp = row.get(f"bid_price_{i}")
            bs = row.get(f"bid_size_{i}")
            ap = row.get(f"ask_price_{i}")
            a_s = row.get(f"ask_size_{i}")

            if bp and bp > 0 and bs and bs > 0:
                bid_prices.append(float(bp))
                bid_sizes.append(float(bs))
            if ap and ap > 0 and a_s and a_s > 0:
                ask_prices.append(float(ap))
                ask_sizes.append(float(a_s))

        bid_cum = np.cumsum(bid_sizes) if bid_sizes else np.array([])
        ask_cum = np.cumsum(ask_sizes) if ask_sizes else np.array([])

        timestamp = row.get("datetime") or row.get("timestamp")

        fig = go.Figure()

        if len(bid_prices) > 0:
            fig.add_trace(
                go.Bar(
                    x=bid_prices,
                    y=bid_sizes,
                    name="Bid (买单)",
                    marker_color=GREEN_COLOR,
                    opacity=0.7,
                    width=0.005,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bid_prices,
                    y=bid_cum,
                    mode="lines+markers",
                    name="Bid 累积深度",
                    line=dict(color=GREEN_COLOR, width=2, shape="hv"),
                    marker=dict(size=5),
                )
            )

        if len(ask_prices) > 0:
            fig.add_trace(
                go.Bar(
                    x=ask_prices,
                    y=ask_sizes,
                    name="Ask (卖单)",
                    marker_color=RED_COLOR,
                    opacity=0.7,
                    width=0.005,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ask_prices,
                    y=ask_cum,
                    mode="lines+markers",
                    name="Ask 累积深度",
                    line=dict(color=RED_COLOR, width=2, shape="hv"),
                    marker=dict(size=5),
                )
            )

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>时间: {timestamp}</sub>",
                x=0.5,
            ),
            xaxis_title="价格",
            yaxis_title="数量 (股)",
            barmode="overlay",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            shapes=[
                dict(
                    type="line",
                    x0=row.get("best_bid", 0),
                    x1=row.get("best_bid", 0),
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color=GREEN_COLOR, width=1, dash="dash"),
                ),
                dict(
                    type="line",
                    x0=row.get("best_ask", 0),
                    x1=row.get("best_ask", 0),
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color=RED_COLOR, width=1, dash="dash"),
                ),
            ],
            annotations=[
                dict(
                    x=row.get("best_bid", 0),
                    y=0.95,
                    yref="paper",
                    text=f"Best Bid: {row.get('best_bid', 0):.4f}",
                    showarrow=False,
                    font=dict(color=GREEN_COLOR),
                    xanchor="right",
                ),
                dict(
                    x=row.get("best_ask", 0),
                    y=0.95,
                    yref="paper",
                    text=f"Best Ask: {row.get('best_ask', 0):.4f}",
                    showarrow=False,
                    font=dict(color=RED_COLOR),
                    xanchor="left",
                ),
            ],
        )

        return _apply_dark_theme(fig)

    def plot_depth_profile(
        self,
        title: str = "平均累计深度剖面",
    ) -> "go.Figure":
        """
        绘制十档平均累计深度剖面图

        Parameters
        ----------
        title : str
            图表标题

        Returns
        -------
        go.Figure
        """
        if self.snapshots_df is None:
            raise ValueError("请先设置快照数据")

        df = self.snapshots_df

        bid_avg_sizes = []
        ask_avg_sizes = []
        levels = list(range(1, 11))

        for i in levels:
            bs = df[f"bid_size_{i}"].drop_nulls().mean() or 0.0
            a_s = df[f"ask_size_{i}"].drop_nulls().mean() or 0.0
            bid_avg_sizes.append(float(bs))
            ask_avg_sizes.append(float(a_s))

        bid_cum = np.cumsum(bid_avg_sizes)
        ask_cum = np.cumsum(ask_avg_sizes)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=levels,
                y=bid_avg_sizes,
                name=f"Bid 档位深度",
                marker_color=GREEN_COLOR,
                opacity=0.7,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=levels,
                y=bid_cum,
                mode="lines+markers",
                name="Bid 累积深度",
                line=dict(color=GREEN_COLOR, width=2),
                marker=dict(size=7),
            )
        )

        fig.add_trace(
            go.Bar(
                x=levels,
                y=ask_avg_sizes,
                name=f"Ask 档位深度",
                marker_color=RED_COLOR,
                opacity=0.7,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=levels,
                y=ask_cum,
                mode="lines+markers",
                name="Ask 累积深度",
                line=dict(color=RED_COLOR, width=2),
                marker=dict(size=7),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="档位 (Level)",
            yaxis_title="平均数量 (股)",
            barmode="group",
            height=450,
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return _apply_dark_theme(fig)

    def plot_spread_distribution(
        self,
        title: str = "价差分布直方图",
        nbins: int = 50,
    ) -> "go.Figure":
        """
        绘制价差分布图

        Parameters
        ----------
        title : str
            图表标题
        nbins : int
            直方图分箱数

        Returns
        -------
        go.Figure
        """
        if self.snapshots_df is None:
            raise ValueError("请先设置快照数据")

        spreads = self.snapshots_df["spread"].drop_nulls().to_list()

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=spreads,
                nbinsx=nbins,
                name="Spread 分布",
                marker_color=BLUE_COLOR,
                opacity=0.8,
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="价差 (Spread)",
            yaxis_title="频次",
            height=400,
            bargap=0.05,
        )

        return _apply_dark_theme(fig)

    def plot_comprehensive_dashboard(
        self,
        stock: str = "TEST",
    ) -> "go.Figure":
        """
        绘制综合微观结构仪表盘

        包含：中间价轨迹、价差时序、订单簿深度、深度剖面 四个子图

        Parameters
        ----------
        stock : str
            股票代码

        Returns
        -------
        go.Figure
        """
        if self.snapshots_df is None:
            raise ValueError("请先设置快照数据")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "中间价 & Best Bid/Ask",
                "价差时间序列",
                "最新十档订单簿深度",
                "平均累计深度剖面",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
        )

        df = self.snapshots_df

        fig.add_trace(
            go.Scatter(
                x=df["datetime"].to_list(),
                y=df["midprice"].to_list(),
                mode="lines",
                name="Midprice",
                line=dict(color=BLUE_COLOR, width=1.2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["datetime"].to_list(),
                y=df["best_bid"].to_list(),
                mode="lines",
                name="Best Bid",
                line=dict(color=GREEN_COLOR, width=0.7, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["datetime"].to_list(),
                y=df["best_ask"].to_list(),
                mode="lines",
                name="Best Ask",
                line=dict(color=RED_COLOR, width=0.7, dash="dash"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["datetime"].to_list(),
                y=df["spread"].to_list(),
                mode="lines",
                name="Spread",
                line=dict(color=BLUE_COLOR, width=1),
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.15)",
            ),
            row=1,
            col=2,
        )

        row = df.row(-1, named=True)
        bid_prices = []
        bid_sizes = []
        ask_prices = []
        ask_sizes = []
        for i in range(1, 11):
            bp = row.get(f"bid_price_{i}")
            bs = row.get(f"bid_size_{i}")
            ap = row.get(f"ask_price_{i}")
            a_s = row.get(f"ask_size_{i}")
            if bp and bp > 0 and bs and bs > 0:
                bid_prices.append(float(bp))
                bid_sizes.append(float(bs))
            if ap and ap > 0 and a_s and a_s > 0:
                ask_prices.append(float(ap))
                ask_sizes.append(float(a_s))

        if bid_prices:
            fig.add_trace(
                go.Bar(
                    x=bid_prices,
                    y=bid_sizes,
                    name="Bid",
                    marker_color=GREEN_COLOR,
                    opacity=0.7,
                    width=0.005,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        if ask_prices:
            fig.add_trace(
                go.Bar(
                    x=ask_prices,
                    y=ask_sizes,
                    name="Ask",
                    marker_color=RED_COLOR,
                    opacity=0.7,
                    width=0.005,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        levels = list(range(1, 11))
        bid_avg = [float(df[f"bid_size_{i}"].drop_nulls().mean() or 0.0) for i in levels]
        ask_avg = [float(df[f"ask_size_{i}"].drop_nulls().mean() or 0.0) for i in levels]
        bid_cum = list(np.cumsum(bid_avg))
        ask_cum = list(np.cumsum(ask_avg))

        fig.add_trace(
            go.Scatter(
                x=levels,
                y=bid_cum,
                mode="lines+markers",
                name="Bid 累积",
                line=dict(color=GREEN_COLOR, width=2),
                marker=dict(size=5),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=levels,
                y=ask_cum,
                mode="lines+markers",
                name="Ask 累积",
                line=dict(color=RED_COLOR, width=2),
                marker=dict(size=5),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_xaxes(title_text="时间", row=1, col=1)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_xaxes(title_text="时间", row=1, col=2)
        fig.update_yaxes(title_text="价差", row=1, col=2)
        fig.update_xaxes(title_text="价格", row=2, col=1)
        fig.update_yaxes(title_text="数量", row=2, col=1)
        fig.update_xaxes(title_text="档位", row=2, col=2, tickmode="linear", tick0=1, dtick=1)
        fig.update_yaxes(title_text="累积数量", row=2, col=2)

        fig.update_layout(
            height=800,
            title=dict(
                text=f"高频微观结构分析仪表盘 - {stock}",
                x=0.5,
                font=dict(size=16),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.005,
                xanchor="right",
                x=1,
            ),
        )

        return _apply_dark_theme(fig)

    @staticmethod
    def save_figure(fig: "go.Figure", filepath: str) -> None:
        """
        保存图表到文件

        Parameters
        ----------
        fig : go.Figure
            Plotly 图表对象
        filepath : str
            文件路径，支持 .html, .png, .svg, .pdf
        """
        if filepath.endswith(".html"):
            fig.write_html(filepath)
        else:
            fig.write_image(filepath)

    def plot_candlestick(
        self,
        ohlc_df: pl.DataFrame,
        title: str = "K 线图",
    ) -> "go.Figure":
        """
        绘制 K 线图（蜡烛图）

        Parameters
        ----------
        ohlc_df : pl.DataFrame
            包含 datetime, open, high, low, close, volume 的 K 线数据
        title : str
            图表标题

        Returns
        -------
        go.Figure
            Plotly 图表对象
        """
        _check_plotly()
        if ohlc_df is None or ohlc_df.is_empty():
            raise ValueError("请提供有效的 OHLC 数据")

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=ohlc_df["datetime"].to_list(),
                open=ohlc_df["open"].to_list(),
                high=ohlc_df["high"].to_list(),
                low=ohlc_df["low"].to_list(),
                close=ohlc_df["close"].to_list(),
                name="价格",
                increasing_line_color=GREEN_COLOR,
                decreasing_line_color=RED_COLOR,
                increasing_fillcolor=f"rgba(34, 197, 94, 0.3)",
                decreasing_fillcolor=f"rgba(239, 68, 68, 0.3)",
            )
        )

        if "volume" in ohlc_df.columns:
            fig.add_trace(
                go.Bar(
                    x=ohlc_df["datetime"].to_list(),
                    y=ohlc_df["volume"].to_list(),
                    name="成交量",
                    marker_color=BLUE_COLOR,
                    opacity=0.4,
                    yaxis="y2",
                )
            )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="时间",
            yaxis_title="价格",
            height=500,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        if "volume" in ohlc_df.columns:
            fig.update_layout(
                yaxis2=dict(
                    title="成交量",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
            )

        return _apply_dark_theme(fig)

    def plot_vpin_with_candlestick(
        self,
        ohlc_df: pl.DataFrame,
        vpin_result: "VPINResult",
        title: str = "价格走势 & VPIN 知情交易概率",
        stock: str = "",
        vpin_threshold: float = 0.5,
    ) -> "go.Figure":
        """
        绘制 K 线主图 + VPIN 知情交易概率副图的组合图表

        VPIN 用于捕捉庄家砸盘前的异常买卖失衡。
        当 VPIN 超过阈值（通常 0.5~0.8）时，表示市场存在异常知情交易，
        可能预示着即将发生大幅单边行情（如砸盘或拉盘）。

        Parameters
        ----------
        ohlc_df : pl.DataFrame
            包含 datetime, open, high, low, close, volume 的 K 线数据
        vpin_result : VPINResult
            compute_vpin() 返回的 VPIN 计算结果
        title : str
            图表标题
        stock : str
            股票代码，用于副标题
        vpin_threshold : float
            VPIN 危险阈值，默认 0.5。超过该值的区域将用红色高亮标注

        Returns
        -------
        go.Figure
            Plotly 组合图表对象
        """
        _check_plotly()
        if ohlc_df is None or ohlc_df.is_empty():
            raise ValueError("请提供有效的 OHLC 数据")
        if vpin_result.vpin_df is None or vpin_result.vpin_df.is_empty():
            raise ValueError("请提供有效的 VPIN 结果")

        vpin_df = vpin_result.vpin_df

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{stock} 价格走势" if stock else "价格走势",
                f"VPIN 知情交易概率 (滚动窗口 {vpin_result.n_buckets if vpin_result.n_buckets < 50 else 50} 桶, 每桶约 {vpin_result.bucket_size:,} 股)",
            ),
            specs=[
                [{"type": "candlestick"}],
                [{"type": "scatter"}],
            ],
            row_heights=[0.65, 0.35],
        )

        fig.add_trace(
            go.Candlestick(
                x=ohlc_df["datetime"].to_list(),
                open=ohlc_df["open"].to_list(),
                high=ohlc_df["high"].to_list(),
                low=ohlc_df["low"].to_list(),
                close=ohlc_df["close"].to_list(),
                name="价格",
                increasing_line_color=GREEN_COLOR,
                decreasing_line_color=RED_COLOR,
                increasing_fillcolor=f"rgba(34, 197, 94, 0.3)",
                decreasing_fillcolor=f"rgba(239, 68, 68, 0.3)",
            ),
            row=1,
            col=1,
        )

        if "volume" in ohlc_df.columns:
            fig.add_trace(
                go.Bar(
                    x=ohlc_df["datetime"].to_list(),
                    y=ohlc_df["volume"].to_list(),
                    name="成交量",
                    marker_color=BLUE_COLOR,
                    opacity=0.3,
                    yaxis="y2",
                ),
                row=1,
                col=1,
            )

        vpin_x = vpin_df["end_datetime"].to_list() if "end_datetime" in vpin_df.columns else vpin_df["start_datetime"].to_list()
        vpin_y = vpin_df["vpin"].to_list()

        fig.add_trace(
            go.Scatter(
                x=vpin_x,
                y=vpin_y,
                mode="lines",
                name="VPIN",
                line=dict(color=BLUE_COLOR, width=2),
                fill="tozeroy",
                fillcolor=f"rgba(59, 130, 246, 0.2)",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=vpin_x,
                y=[vpin_threshold] * len(vpin_y),
                mode="lines",
                name=f"阈值 {vpin_threshold}",
                line=dict(color=RED_COLOR, width=1, dash="dash"),
            ),
            row=2,
            col=1,
        )

        high_vpin_mask = [v >= vpin_threshold for v in vpin_y]
        if any(high_vpin_mask):
            high_x = [vpin_x[i] for i in range(len(vpin_x)) if high_vpin_mask[i]]
            high_y = [vpin_y[i] for i in range(len(vpin_y)) if high_vpin_mask[i]]
            fig.add_trace(
                go.Scatter(
                    x=high_x,
                    y=high_y,
                    mode="markers",
                    name="高风险区域",
                    marker=dict(
                        color=RED_COLOR,
                        size=8,
                        opacity=0.8,
                        line=dict(width=1, color="#ffffff"),
                    ),
                ),
                row=2,
                col=1,
            )

        buy_vol = vpin_df["buy_volume"].to_list()
        sell_vol = vpin_df["sell_volume"].to_list()

        fig.add_trace(
            go.Bar(
                x=vpin_x,
                y=buy_vol,
                name="买方成交量",
                marker_color=GREEN_COLOR,
                opacity=0.25,
                yaxis="y4",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=vpin_x,
                y=[-v for v in sell_vol],
                name="卖方成交量",
                marker_color=RED_COLOR,
                opacity=0.25,
                yaxis="y4",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(
            title_text="时间",
            row=2,
            col=1,
            rangeslider_visible=False,
        )
        fig.update_yaxes(
            title_text="价格",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="VPIN",
            row=2,
            col=1,
            range=[0, max(1.0, float(max(vpin_y)) * 1.1) if vpin_y else [0, 1]],
        )

        if "volume" in ohlc_df.columns:
            fig.update_layout(
                yaxis2=dict(
                    title="成交量",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                yaxis4=dict(
                    title="买卖盘成交量",
                    overlaying="y3",
                    side="right",
                    showgrid=False,
                    showticklabels=False,
                ),
            )

        fig.update_layout(
            height=750,
            title=dict(
                text=f"{title}<br><sub>VPIN 均值: {vpin_result.avg_vpin:.4f}, 峰值: {vpin_result.max_vpin:.4f} | 总量桶: {vpin_result.n_buckets}, 桶容量: {vpin_result.bucket_size:,} 股</sub>",
                x=0.5,
                font=dict(size=14),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.005,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
        )

        return _apply_dark_theme(fig)
