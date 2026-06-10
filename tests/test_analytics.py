"""
单元测试 - 微观结构分析与可视化模块
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hft_lob import ITCHParser, L3OrderBook, MicrostructureAnalytics, MicrostructureViz


def _build_test_data():
    parser = ITCHParser(stock_filter=["TEST"])
    parser.generate_synthetic(stock="TEST", n_messages=50000, seed=42)
    parsed = parser.to_polars()

    ob = L3OrderBook(snapshot_interval_us=100_000, max_snapshots=100000)
    ob.ingest_parsed_messages(parsed)
    ob.build()

    snapshots_df = ob.to_polars()
    return parsed, snapshots_df


def test_microstructure_metrics():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_snapshots(snapshots_df)
    analytics.load_parsed_messages(parsed)

    metrics = analytics.compute_metrics()

    assert metrics.n_snapshots > 0
    assert metrics.avg_spread >= -1  # 允许小的负值（合成数据可能短暂交叉）
    assert metrics.median_spread >= -1
    assert metrics.avg_midprice > 0
    assert metrics.avg_bid_depth_1 >= 0
    assert metrics.avg_ask_depth_1 >= 0
    assert -1.5 <= metrics.order_imbalance <= 1.5

    print(f"  平均价差: {metrics.avg_spread:.6f}")
    print(f"  订单不平衡: {metrics.order_imbalance:.4f}")
    print(f"  年化波动率: {metrics.midprice_volatility:.4f}")
    print("[PASS] test_microstructure_metrics")


def test_resample_snapshots():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_snapshots(snapshots_df)

    resampled = analytics.resample_snapshots(rule="1s")
    assert len(resampled) > 0
    assert "spread_mean" in resampled.columns
    assert "midprice_last" in resampled.columns

    print(f"  原始快照: {len(snapshots_df)}, 1s重采样后: {len(resampled)}")
    print("[PASS] test_resample_snapshots")


def test_order_flow_imbalance():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_snapshots(snapshots_df)

    ofi = analytics.compute_order_flow_imbalance(window_us=10_000_000)
    assert "ofi_sum" in ofi.columns
    assert "midprice" in ofi.columns

    print(f"  OFI 窗口数: {len(ofi)}")
    print("[PASS] test_order_flow_imbalance")


def test_depth_profile():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_snapshots(snapshots_df)

    profile = analytics.compute_depth_profile()
    assert len(profile) == 20
    assert "level" in profile.columns
    assert "side" in profile.columns
    assert "avg_price" in profile.columns
    assert "avg_size" in profile.columns

    print("[PASS] test_depth_profile")


def test_message_statistics():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_parsed_messages(parsed)

    stats = analytics.message_statistics()
    assert len(stats) > 0
    assert "msg_type_name" in stats.columns
    assert "count" in stats.columns

    print(f"  消息类型数: {len(stats)}")
    print("[PASS] test_message_statistics")


def test_visualization_plots():
    parsed, snapshots_df = _build_test_data()

    viz = MicrostructureViz(snapshots_df)

    fig_spread = viz.plot_spread_series()
    assert fig_spread is not None
    assert len(fig_spread.data) > 0

    fig_mid = viz.plot_midprice_series()
    assert fig_mid is not None
    assert len(fig_mid.data) > 0

    fig_depth = viz.plot_order_book_depth()
    assert fig_depth is not None

    fig_profile = viz.plot_depth_profile()
    assert fig_profile is not None
    assert len(fig_profile.data) > 0

    fig_hist = viz.plot_spread_distribution()
    assert fig_hist is not None

    fig_dash = viz.plot_comprehensive_dashboard(stock="TEST")
    assert fig_dash is not None

    print("[PASS] test_visualization_plots")


def test_figure_save():
    parsed, snapshots_df = _build_test_data()

    viz = MicrostructureViz(snapshots_df)
    fig = viz.plot_spread_series()

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "test.html")
        viz.save_figure(fig, html_path)
        assert os.path.exists(html_path)
        assert os.path.getsize(html_path) > 0

    print("[PASS] test_figure_save")


def test_lee_ready_tick_test():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_parsed_messages(parsed)
    analytics.load_snapshots(snapshots_df)

    trades_signed = analytics.lee_ready_tick_test()
    assert len(trades_signed) > 0, "应生成带方向判定的成交数据"
    assert "trade_sign" in trades_signed.columns
    assert "signed_volume" in trades_signed.columns

    signs = trades_signed["trade_sign"].to_list()
    n_buy = sum(1 for s in signs if s > 0)
    n_sell = sum(1 for s in signs if s < 0)
    assert n_buy > 0, "应存在买方发起的成交"
    assert n_sell > 0, "应存在卖方发起的成交"

    print(f"  Lee-Ready 判定: 买方发起 {n_buy} 笔, 卖方发起 {n_sell} 笔")
    print(f"  总成交笔数: {len(trades_signed)}")
    print("[PASS] test_lee_ready_tick_test")


def test_volume_buckets():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_parsed_messages(parsed)
    analytics.load_snapshots(snapshots_df)

    trades_signed = analytics.lee_ready_tick_test()
    buckets = analytics.compute_volume_buckets(trades_signed, n_buckets=20)

    assert len(buckets) > 0, "应生成成交量桶"
    assert "buy_volume" in buckets.columns
    assert "sell_volume" in buckets.columns
    assert "abs_imbalance" in buckets.columns

    total_buy = int(buckets["buy_volume"].sum())
    total_sell = int(buckets["sell_volume"].sum())
    avg_imb = float(buckets["abs_imbalance"].mean())

    print(f"  桶数量: {len(buckets)}")
    print(f"  桶均成交量: {int((total_buy + total_sell) / max(1, len(buckets))):,} 股")
    print(f"  总买方成交量: {total_buy:,} 股")
    print(f"  总卖方成交量: {total_sell:,} 股")
    print(f"  平均失衡量: {avg_imb:.1f} 股")
    print("[PASS] test_volume_buckets")


def test_vpin_computation():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_parsed_messages(parsed)
    analytics.load_snapshots(snapshots_df)

    vpin_result = analytics.compute_vpin(n_buckets=20, vpin_window=10)

    assert vpin_result.n_buckets > 0, "应生成 VPIN 结果"
    assert len(vpin_result.vpin_df) > 0
    assert "vpin" in vpin_result.vpin_df.columns
    assert 0.0 <= vpin_result.avg_vpin <= 1.0, "VPIN 均值应在 [0,1] 区间"
    assert 0.0 <= vpin_result.max_vpin <= 1.0, "VPIN 峰值应在 [0,1] 区间"

    print(f"  {vpin_result}")
    print(f"  VPIN 区间: [{float(vpin_result.vpin_df['vpin'].min()):.4f}, {vpin_result.max_vpin:.4f}]")
    print("[PASS] test_vpin_computation")


def test_candlestick_aggregation():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_parsed_messages(parsed)
    analytics.load_snapshots(snapshots_df)

    ohlc = analytics.compute_candlesticks_from_snapshots(rule="1s")
    assert len(ohlc) > 0, "应生成 K 线数据"
    assert "open" in ohlc.columns
    assert "high" in ohlc.columns
    assert "low" in ohlc.columns
    assert "close" in ohlc.columns
    assert "volume" in ohlc.columns

    for i in range(len(ohlc)):
        row = ohlc.row(i, named=True)
        if row["high"] is not None and row["low"] is not None:
            assert row["high"] >= row["low"], "最高价应不低于最低价"
            if row["open"] is not None:
                assert row["high"] >= row["open"], "最高价应不低于开盘价"
                assert row["low"] <= row["open"], "最低价应不高于开盘价"

    print(f"  K 线数量: {len(ohlc)}")
    print(f"  总成交量: {int(ohlc['volume'].sum() or 0):,} 股")
    print("[PASS] test_candlestick_aggregation")


def test_vpin_visualization():
    parsed, snapshots_df = _build_test_data()

    analytics = MicrostructureAnalytics()
    analytics.load_parsed_messages(parsed)
    analytics.load_snapshots(snapshots_df)

    vpin_result = analytics.compute_vpin(n_buckets=20, vpin_window=10)
    ohlc = analytics.compute_candlesticks_from_snapshots(rule="1s")

    viz = MicrostructureViz(snapshots_df)

    fig_candle = viz.plot_candlestick(ohlc)
    assert fig_candle is not None
    assert len(fig_candle.data) >= 1

    fig_vpin = viz.plot_vpin_with_candlestick(
        ohlc_df=ohlc,
        vpin_result=vpin_result,
        stock="TEST",
    )
    assert fig_vpin is not None
    assert len(fig_vpin.data) >= 3  # K线 + VPIN + 阈值线

    print(f"  K线图 traces: {len(fig_candle.data)}")
    print(f"  VPIN组合图 traces: {len(fig_vpin.data)}")
    print("[PASS] test_vpin_visualization")


if __name__ == "__main__":
    test_microstructure_metrics()
    test_resample_snapshots()
    test_order_flow_imbalance()
    test_depth_profile()
    test_message_statistics()
    test_lee_ready_tick_test()
    test_volume_buckets()
    test_vpin_computation()
    test_candlestick_aggregation()
    test_visualization_plots()
    test_vpin_visualization()
    test_figure_save()
    print("\n所有分析与可视化测试通过!")
