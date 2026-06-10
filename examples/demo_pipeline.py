"""
HFT LOB 微观结构分析平台 - 综合示例
演示从 ITCH 数据解析 -> 订单簿重构 -> 微观结构分析 -> 可视化的完整管线
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hft_lob import ITCHParser, L3OrderBook, MicrostructureAnalytics, MicrostructureViz


def main():
    print("=" * 70)
    print("HFT LOB 微观结构分析平台 - 综合演示")
    print("=" * 70)

    stock = "TEST"
    n_messages = 100_000
    snapshot_interval = 100_000

    print(f"\n[1/6] 生成合成 ITCH 数据: {n_messages:,} 条消息 ...")
    t0 = time.time()
    parser = ITCHParser(stock_filter=[stock])
    parser.generate_synthetic(stock=stock, n_messages=n_messages, seed=42)
    t1 = time.time()
    parsed = parser.to_polars()
    print(f"  解析完成，耗时 {t1 - t0:.3f}s")
    for name, df in parsed.items():
        print(f"    {name}: {len(df):,} 条")

    print(f"\n[2/6] L3 订单簿重构 (Numba JIT 加速) ...")
    print(f"  快照间隔: {snapshot_interval:,} 微秒 ({snapshot_interval/1_000_000:.3f}s)")
    t0 = time.time()
    ob = L3OrderBook(snapshot_interval_us=snapshot_interval, max_snapshots=1_000_000)
    ob.ingest_parsed_messages(parsed)
    stats = ob.build()
    t1 = time.time()
    snapshots_df = ob.to_polars()
    print(f"  重构完成，耗时 {t1 - t0:.3f}s")
    print(f"  {stats}")
    print(f"  生成快照: {len(snapshots_df):,} 个")
    if len(snapshots_df) > 0:
        print(f"  首个快照 Best Bid/Ask: {snapshots_df['best_bid'][0]:.4f} / {snapshots_df['best_ask'][0]:.4f}")
        print(f"  最后快照 Best Bid/Ask: {snapshots_df['best_bid'][-1]:.4f} / {snapshots_df['best_ask'][-1]:.4f}")
        ts, spread = ob.spread_series()
        print(f"  平均价差: {spread.mean():.6f}")
        print(f"  价差标准差: {spread.std():.6f}")

    print(f"\n[3/6] 微观结构指标计算 ...")
    t0 = time.time()
    analytics = MicrostructureAnalytics()
    analytics.load_snapshots(snapshots_df)
    analytics.load_parsed_messages(parsed)
    metrics = analytics.compute_metrics()
    t1 = time.time()
    print(f"  计算完成，耗时 {t1 - t0:.3f}s")
    print(f"    平均价差        : {metrics.avg_spread:.6f}")
    print(f"    价差中位数      : {metrics.median_spread:.6f}")
    print(f"    价差波动率      : {metrics.spread_std:.6f}")
    print(f"    平均中间价      : {metrics.avg_midprice:.4f}")
    print(f"    中间价年化波动率: {metrics.midprice_volatility:.4f}")
    print(f"    平均买一深度    : {metrics.avg_bid_depth_1:.1f}")
    print(f"    平均卖一深度    : {metrics.avg_ask_depth_1:.1f}")
    print(f"    十档累计买深度  : {metrics.total_depth_bid:.1f}")
    print(f"    十档累计卖深度  : {metrics.total_depth_ask:.1f}")
    print(f"    订单不平衡度    : {metrics.order_imbalance:.4f}")

    print(f"\n[4/6] 订单流不平衡 (OFI) 指标 ...")
    ofi_df = analytics.compute_order_flow_imbalance(window_us=10_000_000)
    print(f"  OFI 样本数: {len(ofi_df)}")
    if len(ofi_df) > 0:
        print(f"  OFI 均值: {ofi_df['ofi_sum'].mean():.1f}")

    print(f"\n[5/6] 生成可视化图表 ...")
    viz = MicrostructureViz(snapshots_df)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    fig_spread = viz.plot_spread_series(title=f"{stock} Bid-Ask Spread")
    spread_path = os.path.join(output_dir, f"{stock}_spread.html")
    viz.save_figure(fig_spread, spread_path)
    print(f"  价差时序图 -> {spread_path}")

    fig_midprice = viz.plot_midprice_series(title=f"{stock} Midprice 轨迹")
    mid_path = os.path.join(output_dir, f"{stock}_midprice.html")
    viz.save_figure(fig_midprice, mid_path)
    print(f"  中间价轨迹图 -> {mid_path}")

    fig_depth = viz.plot_order_book_depth(title=f"{stock} L3 订单簿十档深度")
    depth_path = os.path.join(output_dir, f"{stock}_orderbook_depth.html")
    viz.save_figure(fig_depth, depth_path)
    print(f"  订单簿深度图 -> {depth_path}")

    fig_profile = viz.plot_depth_profile(title=f"{stock} 平均累计深度剖面")
    profile_path = os.path.join(output_dir, f"{stock}_depth_profile.html")
    viz.save_figure(fig_profile, profile_path)
    print(f"  深度剖面图 -> {profile_path}")

    fig_hist = viz.plot_spread_distribution(title=f"{stock} 价差分布")
    hist_path = os.path.join(output_dir, f"{stock}_spread_histogram.html")
    viz.save_figure(fig_hist, hist_path)
    print(f"  价差分布图 -> {hist_path}")

    fig_dashboard = viz.plot_comprehensive_dashboard(stock=stock)
    dash_path = os.path.join(output_dir, f"{stock}_dashboard.html")
    viz.save_figure(fig_dashboard, dash_path)
    print(f"  综合仪表盘 -> {dash_path}")

    print(f"\n[6/6] VPIN 知情交易概率分析 ...")
    t0 = time.time()
    analytics.load_snapshots(snapshots_df)
    analytics.load_parsed_messages(parsed)

    vpin_result = analytics.compute_vpin(n_buckets=200, vpin_window=50)
    t1 = time.time()
    print(f"  VPIN 计算完成，耗时 {t1 - t0:.3f}s")
    print(f"  {vpin_result}")

    ohlc_df = analytics.compute_candlesticks_from_snapshots(rule="1s")
    print(f"  K 线数量: {len(ohlc_df)}")

    fig_vpin = viz.plot_vpin_with_candlestick(
        ohlc_df=ohlc_df,
        vpin_result=vpin_result,
        title=f"{stock} 价格 & VPIN 知情交易概率",
        stock=stock,
        vpin_threshold=0.5,
    )
    vpin_path = os.path.join(output_dir, f"{stock}_vpin_candlestick.html")
    viz.save_figure(fig_vpin, vpin_path)
    print(f"  VPIN + K线组合图 -> {vpin_path}")

    trades_signed = vpin_result.trades_with_sign
    if len(trades_signed) > 0:
        n_buy = int((trades_signed["trade_sign"] > 0).sum())
        n_sell = int((trades_signed["trade_sign"] < 0).sum())
        print(f"  Lee-Ready 判定: 买方发起 {n_buy} 笔, 卖方发起 {n_sell} 笔")

    print("\n" + "=" * 70)
    print("全部管线运行完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
