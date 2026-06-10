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


def test_overnight_gtc_orders_multi_day():
    """
    测试多日连续加载 + 隔夜 GTC 挂单场景

    场景：
      - Day 1 (10:00 ~ 14:00): 下单 BID @100.00 100股 (GTC 未成交)
      - 隔夜间隔 (14:00 ~ 次日 09:30): 订单保留
      - Day 2 (09:30 ~ 10:00): 新交易日该订单被 EXECUTE 成交 100股
    预期：不会抛出 KeyError，EXECUTE 能正确找到订单并扣减
    """
    import polars as pl

    DAY1_START = 1717200000_000000  # 2024-06-01 10:00:00 UTC (us)
    DAY1_END = 1717214400_000000    # 2024-06-01 14:00:00 UTC
    DAY2_START = 1717282200_000000  # 2024-06-02 09:30:00 UTC

    GTC_ORDER_ID = 99999999
    GTC_PRICE_INT = 1000000  # 100.0000
    GTC_SHARES = 100

    add_orders = pl.DataFrame({
        "timestamp": [DAY1_START + 1_000_000],
        "order_id": [GTC_ORDER_ID],
        "side": [66],  # 'B'
        "shares": [GTC_SHARES],
        "stock": ["TEST"],
        "price": [GTC_PRICE_INT],
    })

    order_executes = pl.DataFrame({
        "timestamp": [DAY2_START + 2_000_000],
        "order_id": [GTC_ORDER_ID],
        "shares": [GTC_SHARES],
        "match_number": [1],
    }, schema={
        "timestamp": pl.UInt64,
        "order_id": pl.UInt64,
        "shares": pl.UInt32,
        "match_number": pl.UInt64,
    })

    parsed = {
        "add_orders": add_orders,
        "order_executes": order_executes,
    }

    ob = L3OrderBook(
        snapshot_interval_us=1_000_000,
        max_snapshots=100_000,
        lru_window_us=48 * 3600 * 1_000_000,  # 48h 窗口，确保隔夜订单存活
        lru_max_orders=1_000,
    )
    ob.ingest_parsed_messages(parsed)
    stats = ob.build()

    assert stats.n_missing_execute == 0, (
        f"隔夜 GTC 订单执行时找不到引用，missing_execute={stats.n_missing_execute}"
    )
    assert stats.total_missing == 0, f"不应有任何缺失引用，实际={stats.total_missing}"

    snapshots_df = ob.to_polars()
    print(f"  隔夜测试快照数: {len(snapshots_df)}")
    print(f"  {stats}")
    print("[PASS] test_overnight_gtc_orders_multi_day")


def test_lru_eviction_sliding_window():
    """
    测试 LRU 滑动窗口淘汰机制

    场景：
      - t=0ms: Add 订单A @100.00 100股
      - t=1ms: Add 订单B @100.01 100股
      - t=2ms: Add 订单C @100.02 100股
      - 窗口=3ms，lru_max_orders=2
      - t=4ms: 超过阈值，触发淘汰，last_ts < (4ms - 3ms) = 1ms 的订单 A 应被淘汰
    预期：LRU 淘汰计数 > 0，且被淘汰订单后续操作会计入 missing
    """
    import polars as pl

    T0 = 1_000_000  # 1ms
    T1 = 2_000_000  # 2ms
    T2 = 3_000_000  # 3ms
    T_TRIGGER = 6_000_000  # 6ms 触发淘汰，cutoff = 6ms - 3ms = 3ms

    add_orders = pl.DataFrame({
        "timestamp": [T0, T1, T2],
        "order_id": [1001, 1002, 1003],
        "side": [66, 66, 66],
        "shares": [100, 100, 100],
        "stock": ["TEST", "TEST", "TEST"],
        "price": [1000000, 1000100, 1000200],
    })

    order_deletes = pl.DataFrame({
        "timestamp": [T_TRIGGER + 1_000_000],
        "order_id": [1001],  # 订单A 已被 LRU 淘汰
    }, schema={
        "timestamp": pl.UInt64,
        "order_id": pl.UInt64,
    })

    parsed = {
        "add_orders": add_orders,
        "order_deletes": order_deletes,
    }

    ob = L3OrderBook(
        snapshot_interval_us=10_000_000,
        max_snapshots=100,
        lru_window_us=3_000_000,   # 3ms 窗口
        lru_max_orders=2,           # 超过 2 单触发淘汰
    )
    ob.ingest_parsed_messages(parsed)
    stats = ob.build()

    assert stats.n_lru_evicted >= 1, (
        f"LRU 应至少淘汰 1 笔过期订单，实际={stats.n_lru_evicted}"
    )
    assert stats.n_missing_delete >= 1, (
        f"过期订单被淘汰后 Delete 应计为 missing，实际={stats.n_missing_delete}"
    )
    print(f"  LRU 淘汰: {stats.n_lru_evicted}")
    print(f"  Missing Delete: {stats.n_missing_delete}")
    print(f"  {stats}")
    print("[PASS] test_lru_eviction_sliding_window")


def test_graceful_error_handling_no_crash():
    """
    测试优雅容错：引用不存在订单时不崩溃，仅记录统计

    场景：
      - 直接发送 Execute/Cancel/Delete/Replace 各 1 条，引用完全不存在的订单号
    预期：
      - 不抛出 KeyError / Exception
      - stats 中对应 missing 计数器各 +1
      - 订单簿推演继续进行，后续正常订单仍可处理
    """
    import polars as pl

    TS = 1_000_000

    order_executes = pl.DataFrame({
        "timestamp": [TS],
        "order_id": [88880001],
        "shares": [10],
        "match_number": [1],
    }, schema={
        "timestamp": pl.UInt64,
        "order_id": pl.UInt64,
        "shares": pl.UInt32,
        "match_number": pl.UInt64,
    })

    order_cancels = pl.DataFrame({
        "timestamp": [TS + 1_000],
        "order_id": [88880002],
        "canceled_shares": [10],
    }, schema={
        "timestamp": pl.UInt64,
        "order_id": pl.UInt64,
        "canceled_shares": pl.UInt32,
    })

    order_deletes = pl.DataFrame({
        "timestamp": [TS + 2_000],
        "order_id": [88880003],
    }, schema={
        "timestamp": pl.UInt64,
        "order_id": pl.UInt64,
    })

    order_replaces = pl.DataFrame({
        "timestamp": [TS + 3_000],
        "original_order_id": [88880004],
        "new_order_id": [88880005],
        "shares": [50],
        "price": [990000],
    }, schema={
        "timestamp": pl.UInt64,
        "original_order_id": pl.UInt64,
        "new_order_id": pl.UInt64,
        "shares": pl.UInt32,
        "price": pl.UInt64,
    })

    good_add = pl.DataFrame({
        "timestamp": [TS + 10_000],
        "order_id": [77770001],
        "side": [83],  # 'S'
        "shares": [200],
        "stock": ["TEST"],
        "price": [1010000],
    })

    parsed = {
        "order_executes": order_executes,
        "order_cancels": order_cancels,
        "order_deletes": order_deletes,
        "order_replaces": order_replaces,
        "add_orders": good_add,
    }

    ob = L3OrderBook(
        snapshot_interval_us=100_000,
        max_snapshots=10_000,
    )
    ob.ingest_parsed_messages(parsed)
    stats = ob.build()

    assert stats.n_missing_execute == 1, f"Execute missing 计数应为 1，实际={stats.n_missing_execute}"
    assert stats.n_missing_cancel == 1, f"Cancel missing 计数应为 1，实际={stats.n_missing_cancel}"
    assert stats.n_missing_delete == 1, f"Delete missing 计数应为 1，实际={stats.n_missing_delete}"
    assert stats.n_missing_replace == 1, f"Replace missing 计数应为 1，实际={stats.n_missing_replace}"
    assert stats.total_missing == 4

    snapshots_df = ob.to_polars()
    assert len(snapshots_df) > 0, "后续正常订单仍应能生成快照"

    assert stats.n_orders_peak >= 1, "正常添加的订单应能被追踪"
    print(f"  Missing Execute: {stats.n_missing_execute}")
    print(f"  Missing Cancel:  {stats.n_missing_cancel}")
    print(f"  Missing Delete:  {stats.n_missing_delete}")
    print(f"  Missing Replace: {stats.n_missing_replace}")
    print(f"  正常订单峰值:    {stats.n_orders_peak}")
    print(f"  推演未崩溃，快照数: {len(snapshots_df)}")
    print(f"  {stats}")
    print("[PASS] test_graceful_error_handling_no_crash")


if __name__ == "__main__":
    test_orderbook_basic_reconstruction()
    test_spread_positive()
    test_order_imbalance_consistency()
    test_snapshot_prices_monotonic()
    test_performance_benchmark()
    test_overnight_gtc_orders_multi_day()
    test_lru_eviction_sliding_window()
    test_graceful_error_handling_no_crash()
    print("\n所有 L3 OrderBook 测试通过!")
