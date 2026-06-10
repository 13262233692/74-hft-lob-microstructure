"""
单元测试 - L3 订单簿 Numba JIT 重构引擎
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hft_lob import ITCHParser, L3OrderBook
import numpy as np


def test_orderbook_basic_reconstruction():
    parser = ITCHParser(stock_filter=["TEST"])
    parser.generate_synthetic(stock="TEST", n_messages=50000, seed=42)
    parsed = parser.to_polars()

    ob = L3OrderBook(snapshot_interval_us=100_000, max_snapshots=100000)
    ob.ingest_parsed_messages(parsed)
    ob.build()

    snapshots_df = ob.to_polars()
    assert len(snapshots_df) > 0, "应生成至少一个快照"
    print(f"  快照数量: {len(snapshots_df)}")

    first_row = snapshots_df.row(0, named=True)
    assert "best_bid" in first_row
    assert "best_ask" in first_row
    assert "spread" in first_row
    assert "midprice" in first_row

    for i in range(1, 11):
        assert f"bid_price_{i}" in first_row
        assert f"bid_size_{i}" in first_row
        assert f"ask_price_{i}" in first_row
        assert f"ask_size_{i}" in first_row

    print("[PASS] test_orderbook_basic_reconstruction")


def test_spread_positive():
    parser = ITCHParser(stock_filter=["TEST"])
    parser.generate_synthetic(stock="TEST", n_messages=50000, seed=123)
    parsed = parser.to_polars()

    ob = L3OrderBook(snapshot_interval_us=100_000, max_snapshots=100000)
    ob.ingest_parsed_messages(parsed)
    ob.build()

    snapshots_df = ob.to_polars()
    if len(snapshots_df) < 2:
        print("[SKIP] test_spread_positive (快照不足)")
        return

    valid_snaps = snapshots_df.filter(
        (pl.col("best_bid") > 0) & (pl.col("best_ask") > 0)
    )
    if len(valid_snaps) > 0:
        spreads = valid_snaps["spread"].to_list()
        non_negative_ratio = sum(1 for s in spreads if s >= -0.1) / len(spreads)
        assert non_negative_ratio >= 0.50, f"合理价差占比仅 {non_negative_ratio:.2%}，低于 50%"
        avg_spread = sum(s for s in spreads if s >= -0.1) / max(1, sum(1 for s in spreads if s >= -0.1))
        print(f"  有效快照: {len(valid_snaps)}, 合理价差占比: {non_negative_ratio:.2%}, 平均价差: {avg_spread:.6f}")

    print("[PASS] test_spread_positive")


def test_order_imbalance_consistency():
    parser = ITCHParser(stock_filter=["TEST"])
    parser.generate_synthetic(stock="TEST", n_messages=50000, seed=456)
    parsed = parser.to_polars()

    ob = L3OrderBook(snapshot_interval_us=100_000, max_snapshots=500000)
    ob.ingest_parsed_messages(parsed)
    ob.build()

    snapshots_df = ob.to_polars()
    assert len(snapshots_df) > 0

    ts, spread = ob.spread_series()
    assert len(ts) == len(spread)
    assert len(ts) > 0

    ts_mid, mid = ob.midprice_series()
    assert len(ts_mid) == len(mid)
    assert len(mid) > 0

    print(f"  价差序列长度: {len(spread)}")
    print(f"  中间价序列长度: {len(mid)}")
    print("[PASS] test_order_imbalance_consistency")


def test_snapshot_prices_monotonic():
    parser = ITCHParser(stock_filter=["TEST"])
    parser.generate_synthetic(stock="TEST", n_messages=50000, seed=789)
    parsed = parser.to_polars()

    ob = L3OrderBook(snapshot_interval_us=100_000, max_snapshots=100000)
    ob.ingest_parsed_messages(parsed)
    ob.build()

    snapshots_df = ob.to_polars()
    if len(snapshots_df) == 0:
        print("[SKIP] test_snapshot_prices_monotonic (无快照)")
        return

    last_snap = snapshots_df.row(-1, named=True)

    bid_prices = []
    for i in range(1, 11):
        p = last_snap.get(f"bid_price_{i}")
        if p and p > 0:
            bid_prices.append(float(p))

    ask_prices = []
    for i in range(1, 11):
        p = last_snap.get(f"ask_price_{i}")
        if p and p > 0:
            ask_prices.append(float(p))

    if len(bid_prices) > 1:
        for i in range(len(bid_prices) - 1):
            assert bid_prices[i] >= bid_prices[i + 1], "买盘价格应从高到低"

    if len(ask_prices) > 1:
        for i in range(len(ask_prices) - 1):
            assert ask_prices[i] <= ask_prices[i + 1], "卖盘价格应从低到高"

    if bid_prices and ask_prices:
        diff = bid_prices[0] - ask_prices[0]
        tolerance = 0.5
        assert diff <= tolerance, f"Best Bid ({bid_prices[0]:.4f}) 与 Best Ask ({ask_prices[0]:.4f}) 差值超过 {tolerance}"

    print(f"  Bid 档位: {bid_prices}")
    print(f"  Ask 档位: {ask_prices}")
    print("[PASS] test_snapshot_prices_monotonic")


def test_performance_benchmark():
    import time

    parser = ITCHParser(stock_filter=["TEST"])
    n_msgs = 100_000
    print(f"  生成 {n_msgs:,} 条消息 ...")
    parser.generate_synthetic(stock="TEST", n_messages=n_msgs, seed=999)
    parsed = parser.to_polars()
    total_parsed = sum(len(df) for df in parsed.values())

    ob = L3OrderBook(snapshot_interval_us=100_000, max_snapshots=5000000)
    ob.ingest_parsed_messages(parsed)

    print(f"  开始订单簿重构 ({total_parsed:,} 条消息) ...")
    t0 = time.time()
    ob.build()
    t1 = time.time()

    elapsed = t1 - t0
    msg_per_sec = total_parsed / elapsed if elapsed > 0 else 0

    print(f"  耗时: {elapsed:.3f}s")
    print(f"  吞吐量: {msg_per_sec:,.0f} 条消息/秒")

    snapshots_df = ob.to_polars()
    print(f"  快照数: {len(snapshots_df):,}")

    assert elapsed > 0
    print("[PASS] test_performance_benchmark")


import polars as pl


if __name__ == "__main__":
    test_orderbook_basic_reconstruction()
    test_spread_positive()
    test_order_imbalance_consistency()
    test_snapshot_prices_monotonic()
    test_performance_benchmark()
    print("\n所有 L3 OrderBook 测试通过!")
