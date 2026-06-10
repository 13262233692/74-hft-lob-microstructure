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


if __name__ == "__main__":
    test_microstructure_metrics()
    test_resample_snapshots()
    test_order_flow_imbalance()
    test_depth_profile()
    test_message_statistics()
    test_visualization_plots()
    test_figure_save()
    print("\n所有分析与可视化测试通过!")
