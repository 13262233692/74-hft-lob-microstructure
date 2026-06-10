"""
L3 逐笔订单簿 (Level 3 Order Book) 高频重构引擎
利用 Numba JIT 编译实现高性能状态机，在内存中还原全量限价订单簿
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, uint8, uint64, uint32, float64, int64, boolean
from numba.core import types
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList

try:
    import polars as pl
except ImportError:
    pl = None


BUY_SIDE = np.uint8(66)
SELL_SIDE = np.uint8(83)

MSG_ADD = np.uint8(0)
MSG_EXECUTE = np.uint8(1)
MSG_CANCEL = np.uint8(2)
MSG_DELETE = np.uint8(3)
MSG_REPLACE = np.uint8(4)

DEPTH_LEVELS = 10


@njit
def _init_order_dict():
    d = NumbaDict.empty(
        key_type=uint64,
        value_type=types.Tuple([uint32, uint64, uint8]),
    )
    return d


@njit
def _init_price_level_dict():
    d = NumbaDict.empty(
        key_type=uint64,
        value_type=uint64,
    )
    return d


@njit
def _get_top_bid(bid_prices, bid_sizes, n_bids):
    if n_bids == 0:
        return uint64(0), uint64(0)
    idx = 0
    max_p = bid_prices[0]
    for i in range(1, n_bids):
        if bid_prices[i] > max_p:
            max_p = bid_prices[i]
            idx = i
    return bid_prices[idx], bid_sizes[idx]


@njit
def _get_top_ask(ask_prices, ask_sizes, n_asks):
    if n_asks == 0:
        return uint64(0), uint64(0)
    idx = 0
    min_p = ask_prices[0]
    for i in range(1, n_asks):
        if ask_prices[i] < min_p:
            min_p = ask_prices[i]
            idx = i
    return ask_prices[idx], ask_sizes[idx]


@njit
def _get_level_depth(
    side_prices, side_sizes, n_side, depth_levels, is_bid
):
    out_prices = np.zeros(depth_levels, dtype=np.uint64)
    out_sizes = np.zeros(depth_levels, dtype=np.uint64)

    if n_side == 0:
        return out_prices, out_sizes

    prices_copy = np.copy(side_prices[:n_side])
    sizes_copy = np.copy(side_sizes[:n_side])

    n_valid = 0
    for i in range(n_side):
        if sizes_copy[i] > 0:
            n_valid += 1

    if n_valid == 0:
        return out_prices, out_sizes

    n_levels = min(depth_levels, n_valid)

    for lvl in range(n_levels):
        best_idx = -1
        best_p = uint64(0)
        for i in range(len(prices_copy)):
            if sizes_copy[i] == 0:
                continue
            if best_idx < 0:
                best_idx = i
                best_p = prices_copy[i]
            else:
                if is_bid:
                    if prices_copy[i] > best_p:
                        best_p = prices_copy[i]
                        best_idx = i
                else:
                    if prices_copy[i] < best_p:
                        best_p = prices_copy[i]
                        best_idx = i

        if best_idx >= 0 and sizes_copy[best_idx] > 0:
            out_prices[lvl] = prices_copy[best_idx]
            out_sizes[lvl] = sizes_copy[best_idx]
            sizes_copy[best_idx] = uint64(0)

    return out_prices, out_sizes


@njit
def _process_messages_numba(
    msg_types,
    timestamps,
    order_ids,
    sides,
    shares,
    prices,
    new_order_ids,
    snapshot_interval,
    max_snapshots,
):
    """
    Numba JIT 编译的订单簿状态机核心

    使用数组而非字典存储订单，以获得最佳内存访问局部性和性能。
    """
    MAX_ORDERS = max(1000000, len(msg_types) * 2)
    MAX_LEVELS = max(100000, len(msg_types))

    order_map_id = np.zeros(MAX_ORDERS, dtype=np.uint64)
    order_map_shares = np.zeros(MAX_ORDERS, dtype=np.uint64)
    order_map_price = np.zeros(MAX_ORDERS, dtype=np.uint64)
    order_map_side = np.zeros(MAX_ORDERS, dtype=np.uint8)
    order_map_used = np.zeros(MAX_ORDERS, dtype=np.bool_)

    n_orders = 0
    id_to_idx = NumbaDict.empty(key_type=uint64, value_type=uint64)

    buy_price_levels = np.zeros(MAX_LEVELS, dtype=np.uint64)
    buy_size_levels = np.zeros(MAX_LEVELS, dtype=np.uint64)
    buy_level_used = np.zeros(MAX_LEVELS, dtype=np.bool_)
    n_buy_levels = 0

    sell_price_levels = np.zeros(MAX_LEVELS, dtype=np.uint64)
    sell_size_levels = np.zeros(MAX_LEVELS, dtype=np.uint64)
    sell_level_used = np.zeros(MAX_LEVELS, dtype=np.bool_)
    n_sell_levels = 0

    price_to_buy_idx = NumbaDict.empty(key_type=uint64, value_type=uint64)
    price_to_sell_idx = NumbaDict.empty(key_type=uint64, value_type=uint64)

    n_msgs = len(msg_types)

    snap_timestamps = np.zeros(max_snapshots, dtype=np.uint64)
    snap_bid_prices = np.zeros((max_snapshots, DEPTH_LEVELS), dtype=np.uint64)
    snap_bid_sizes = np.zeros((max_snapshots, DEPTH_LEVELS), dtype=np.uint64)
    snap_ask_prices = np.zeros((max_snapshots, DEPTH_LEVELS), dtype=np.uint64)
    snap_ask_sizes = np.zeros((max_snapshots, DEPTH_LEVELS), dtype=np.uint64)
    snap_spread = np.zeros(max_snapshots, dtype=np.float64)
    snap_midprice = np.zeros(max_snapshots, dtype=np.float64)
    n_snapshots = 0

    last_snapshot_ts = uint64(0)

    for i in range(n_msgs):
        mtype = msg_types[i]
        ts = timestamps[i]
        oid = order_ids[i]

        if mtype == MSG_ADD:
            side = sides[i]
            shr = uint64(shares[i])
            prc = prices[i]

            order_map_id[n_orders] = oid
            order_map_shares[n_orders] = shr
            order_map_price[n_orders] = prc
            order_map_side[n_orders] = side
            order_map_used[n_orders] = True
            id_to_idx[oid] = uint64(n_orders)
            n_orders += 1

            if side == BUY_SIDE:
                if prc in price_to_buy_idx:
                    li = price_to_buy_idx[prc]
                    buy_size_levels[li] += shr
                else:
                    buy_price_levels[n_buy_levels] = prc
                    buy_size_levels[n_buy_levels] = shr
                    buy_level_used[n_buy_levels] = True
                    price_to_buy_idx[prc] = uint64(n_buy_levels)
                    n_buy_levels += 1
            else:
                if prc in price_to_sell_idx:
                    li = price_to_sell_idx[prc]
                    sell_size_levels[li] += shr
                else:
                    sell_price_levels[n_sell_levels] = prc
                    sell_size_levels[n_sell_levels] = shr
                    sell_level_used[n_sell_levels] = True
                    price_to_sell_idx[prc] = uint64(n_sell_levels)
                    n_sell_levels += 1

        elif mtype == MSG_EXECUTE:
            shr = uint64(shares[i])
            if oid in id_to_idx:
                oi = id_to_idx[oid]
                if not order_map_used[oi]:
                    continue
                side = order_map_side[oi]
                prc = order_map_price[oi]
                old_shares = order_map_shares[oi]
                new_shares = old_shares - shr
                if new_shares <= 0:
                    order_map_used[oi] = False
                    del id_to_idx[oid]
                    new_shares = uint64(0)
                order_map_shares[oi] = new_shares

                if side == BUY_SIDE:
                    if prc in price_to_buy_idx:
                        li = price_to_buy_idx[prc]
                        buy_size_levels[li] -= shr
                        if buy_size_levels[li] <= 0:
                            buy_size_levels[li] = uint64(0)
                            buy_level_used[li] = False
                            del price_to_buy_idx[prc]
                else:
                    if prc in price_to_sell_idx:
                        li = price_to_sell_idx[prc]
                        sell_size_levels[li] -= shr
                        if sell_size_levels[li] <= 0:
                            sell_size_levels[li] = uint64(0)
                            sell_level_used[li] = False
                            del price_to_sell_idx[prc]

        elif mtype == MSG_CANCEL:
            shr = uint64(shares[i])
            if oid in id_to_idx:
                oi = id_to_idx[oid]
                if not order_map_used[oi]:
                    continue
                side = order_map_side[oi]
                prc = order_map_price[oi]
                old_shares = order_map_shares[oi]
                new_shares = old_shares - shr
                if new_shares <= 0:
                    order_map_used[oi] = False
                    del id_to_idx[oid]
                    new_shares = uint64(0)
                order_map_shares[oi] = new_shares

                if side == BUY_SIDE:
                    if prc in price_to_buy_idx:
                        li = price_to_buy_idx[prc]
                        buy_size_levels[li] -= shr
                        if buy_size_levels[li] <= 0:
                            buy_size_levels[li] = uint64(0)
                            buy_level_used[li] = False
                            del price_to_buy_idx[prc]
                else:
                    if prc in price_to_sell_idx:
                        li = price_to_sell_idx[prc]
                        sell_size_levels[li] -= shr
                        if sell_size_levels[li] <= 0:
                            sell_size_levels[li] = uint64(0)
                            sell_level_used[li] = False
                            del price_to_sell_idx[prc]

        elif mtype == MSG_DELETE:
            if oid in id_to_idx:
                oi = id_to_idx[oid]
                if not order_map_used[oi]:
                    continue
                side = order_map_side[oi]
                prc = order_map_price[oi]
                shr = order_map_shares[oi]
                order_map_used[oi] = False
                del id_to_idx[oid]

                if side == BUY_SIDE:
                    if prc in price_to_buy_idx:
                        li = price_to_buy_idx[prc]
                        buy_size_levels[li] -= shr
                        if buy_size_levels[li] <= 0:
                            buy_size_levels[li] = uint64(0)
                            buy_level_used[li] = False
                            del price_to_buy_idx[prc]
                else:
                    if prc in price_to_sell_idx:
                        li = price_to_sell_idx[prc]
                        sell_size_levels[li] -= shr
                        if sell_size_levels[li] <= 0:
                            sell_size_levels[li] = uint64(0)
                            sell_level_used[li] = False
                            del price_to_sell_idx[prc]

        elif mtype == MSG_REPLACE:
            new_oid = new_order_ids[i]
            side = uint8(0)
            old_prc = uint64(0)
            old_shr = uint64(0)

            if oid in id_to_idx:
                oi = id_to_idx[oid]
                if order_map_used[oi]:
                    side = order_map_side[oi]
                    old_prc = order_map_price[oi]
                    old_shr = order_map_shares[oi]
                    order_map_used[oi] = False
                    del id_to_idx[oid]

            new_shr = uint64(shares[i])
            new_prc = prices[i]

            order_map_id[n_orders] = new_oid
            order_map_shares[n_orders] = new_shr
            order_map_price[n_orders] = new_prc
            order_map_side[n_orders] = side
            order_map_used[n_orders] = True
            id_to_idx[new_oid] = uint64(n_orders)
            n_orders += 1

            if side > 0 and old_shr > 0:
                if side == BUY_SIDE:
                    if old_prc in price_to_buy_idx:
                        li = price_to_buy_idx[old_prc]
                        buy_size_levels[li] -= old_shr
                        if buy_size_levels[li] <= 0:
                            buy_size_levels[li] = uint64(0)
                            buy_level_used[li] = False
                            del price_to_buy_idx[old_prc]

                    if new_prc in price_to_buy_idx:
                        li = price_to_buy_idx[new_prc]
                        buy_size_levels[li] += new_shr
                    else:
                        buy_price_levels[n_buy_levels] = new_prc
                        buy_size_levels[n_buy_levels] = new_shr
                        buy_level_used[n_buy_levels] = True
                        price_to_buy_idx[new_prc] = uint64(n_buy_levels)
                        n_buy_levels += 1
                else:
                    if old_prc in price_to_sell_idx:
                        li = price_to_sell_idx[old_prc]
                        sell_size_levels[li] -= old_shr
                        if sell_size_levels[li] <= 0:
                            sell_size_levels[li] = uint64(0)
                            sell_level_used[li] = False
                            del price_to_sell_idx[old_prc]

                    if new_prc in price_to_sell_idx:
                        li = price_to_sell_idx[new_prc]
                        sell_size_levels[li] += new_shr
                    else:
                        sell_price_levels[n_sell_levels] = new_prc
                        sell_size_levels[n_sell_levels] = new_shr
                        sell_level_used[n_sell_levels] = True
                        price_to_sell_idx[new_prc] = uint64(n_sell_levels)
                        n_sell_levels += 1

        if snapshot_interval > 0 and ts - last_snapshot_ts >= snapshot_interval:
            if n_snapshots < max_snapshots:
                snap_timestamps[n_snapshots] = ts

                active_buy_prices = NumbaList.empty_list(uint64)
                active_buy_sizes = NumbaList.empty_list(uint64)
                for li in range(n_buy_levels):
                    if buy_level_used[li] and buy_size_levels[li] > 0:
                        active_buy_prices.append(buy_price_levels[li])
                        active_buy_sizes.append(buy_size_levels[li])

                active_ask_prices = NumbaList.empty_list(uint64)
                active_ask_sizes = NumbaList.empty_list(uint64)
                for li in range(n_sell_levels):
                    if sell_level_used[li] and sell_size_levels[li] > 0:
                        active_ask_prices.append(sell_price_levels[li])
                        active_ask_sizes.append(sell_size_levels[li])

                n_b = len(active_buy_prices)
                bid_p_arr = np.zeros(max(n_b, 1), dtype=np.uint64)
                bid_s_arr = np.zeros(max(n_b, 1), dtype=np.uint64)
                for k in range(n_b):
                    bid_p_arr[k] = active_buy_prices[k]
                    bid_s_arr[k] = active_buy_sizes[k]

                n_a = len(active_ask_prices)
                ask_p_arr = np.zeros(max(n_a, 1), dtype=np.uint64)
                ask_s_arr = np.zeros(max(n_a, 1), dtype=np.uint64)
                for k in range(n_a):
                    ask_p_arr[k] = active_ask_prices[k]
                    ask_s_arr[k] = active_ask_sizes[k]

                bp, bs = _get_level_depth(bid_p_arr, bid_s_arr, n_b, DEPTH_LEVELS, True)
                ap, a_s = _get_level_depth(ask_p_arr, ask_s_arr, n_a, DEPTH_LEVELS, False)

                for lvl in range(DEPTH_LEVELS):
                    snap_bid_prices[n_snapshots, lvl] = bp[lvl]
                    snap_bid_sizes[n_snapshots, lvl] = bs[lvl]
                    snap_ask_prices[n_snapshots, lvl] = ap[lvl]
                    snap_ask_sizes[n_snapshots, lvl] = a_s[lvl]

                top_bid = bp[0]
                top_ask = ap[0]
                if top_bid > 0 and top_ask > 0:
                    bid_f = float(top_bid) / 10000.0
                    ask_f = float(top_ask) / 10000.0
                    snap_spread[n_snapshots] = ask_f - bid_f
                    snap_midprice[n_snapshots] = (bid_f + ask_f) / 2.0
                else:
                    snap_spread[n_snapshots] = 0.0
                    snap_midprice[n_snapshots] = 0.0

                n_snapshots += 1
                last_snapshot_ts = ts

    return (
        snap_timestamps[:n_snapshots],
        snap_bid_prices[:n_snapshots],
        snap_bid_sizes[:n_snapshots],
        snap_ask_prices[:n_snapshots],
        snap_ask_sizes[:n_snapshots],
        snap_spread[:n_snapshots],
        snap_midprice[:n_snapshots],
    )


@dataclass
class OrderBookSnapshot:
    """订单簿十档深度快照"""

    timestamp: np.uint64
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray
    spread: float
    midprice: float

    @property
    def best_bid(self) -> float:
        return float(self.bid_prices[0]) / 10000.0 if self.bid_prices[0] > 0 else 0.0

    @property
    def best_ask(self) -> float:
        return float(self.ask_prices[0]) / 10000.0 if self.ask_prices[0] > 0 else 0.0

    @property
    def bid_price_floats(self) -> np.ndarray:
        return self.bid_prices.astype(np.float64) / 10000.0

    @property
    def ask_price_floats(self) -> np.ndarray:
        return self.ask_prices.astype(np.float64) / 10000.0


class L3OrderBook:
    """
    L3 逐笔订单簿重构引擎

    使用 Numba JIT 编译的状态机处理全量 ITCH 消息，
    还原出十档买卖盘深度及价差、中间价等微观结构指标。
    """

    def __init__(self, snapshot_interval_us: int = 1_000_000, max_snapshots: int = 10_000_000):
        """
        Parameters
        ----------
        snapshot_interval_us : int
            快照采样间隔（微秒），默认 1ms
        max_snapshots : int
            最大快照数量（防止内存溢出）
        """
        self.snapshot_interval_us = snapshot_interval_us
        self.max_snapshots = max_snapshots
        self.snapshots: List[OrderBookSnapshot] = []
        self._msg_types: List[np.uint8] = []
        self._timestamps: List[np.uint64] = []
        self._order_ids: List[np.uint64] = []
        self._sides: List[np.uint8] = []
        self._shares: List[np.uint32] = []
        self._prices: List[np.uint64] = []
        self._new_order_ids: List[np.uint64] = []

    def ingest_parsed_messages(self, parsed_data: dict) -> None:
        """
        从 ITCHParser 解析结果导入消息

        Parameters
        ----------
        parsed_data : dict
            ITCHParser.to_polars() 返回的 DataFrame 字典
        """
        self._msg_types.clear()
        self._timestamps.clear()
        self._order_ids.clear()
        self._sides.clear()
        self._shares.clear()
        self._prices.clear()
        self._new_order_ids.clear()

        msgs: List[Tuple] = []

        if "add_orders" in parsed_data:
            df = parsed_data["add_orders"]
            for row in df.iter_rows(named=True):
                msgs.append(
                    (
                        int(row["timestamp"]),
                        MSG_ADD,
                        int(row["order_id"]),
                        int(row["side"]),
                        int(row["shares"]),
                        int(row["price"]),
                        0,
                    )
                )

        if "order_executes" in parsed_data:
            df = parsed_data["order_executes"]
            for row in df.iter_rows(named=True):
                msgs.append(
                    (
                        int(row["timestamp"]),
                        MSG_EXECUTE,
                        int(row["order_id"]),
                        0,
                        int(row["shares"]),
                        0,
                        0,
                    )
                )

        if "order_cancels" in parsed_data:
            df = parsed_data["order_cancels"]
            for row in df.iter_rows(named=True):
                msgs.append(
                    (
                        int(row["timestamp"]),
                        MSG_CANCEL,
                        int(row["order_id"]),
                        0,
                        int(row["canceled_shares"]),
                        0,
                        0,
                    )
                )

        if "order_deletes" in parsed_data:
            df = parsed_data["order_deletes"]
            for row in df.iter_rows(named=True):
                msgs.append(
                    (
                        int(row["timestamp"]),
                        MSG_DELETE,
                        int(row["order_id"]),
                        0,
                        0,
                        0,
                        0,
                    )
                )

        if "order_replaces" in parsed_data:
            df = parsed_data["order_replaces"]
            for row in df.iter_rows(named=True):
                msgs.append(
                    (
                        int(row["timestamp"]),
                        MSG_REPLACE,
                        int(row["original_order_id"]),
                        0,
                        int(row["shares"]),
                        int(row["price"]),
                        int(row["new_order_id"]),
                    )
                )

        msgs.sort(key=lambda x: x[0])

        for ts, mtype, oid, side, shr, prc, new_oid in msgs:
            self._timestamps.append(np.uint64(ts))
            self._msg_types.append(np.uint8(mtype))
            self._order_ids.append(np.uint64(oid))
            self._sides.append(np.uint8(side))
            self._shares.append(np.uint32(shr))
            self._prices.append(np.uint64(prc))
            self._new_order_ids.append(np.uint64(new_oid))

    def build(self) -> None:
        """
        执行 Numba JIT 加速的订单簿重建

        处理所有已导入的消息，生成十档深度快照序列。
        """
        if not self._msg_types:
            return

        msg_types = np.array(self._msg_types, dtype=np.uint8)
        timestamps = np.array(self._timestamps, dtype=np.uint64)
        order_ids = np.array(self._order_ids, dtype=np.uint64)
        sides = np.array(self._sides, dtype=np.uint8)
        shares = np.array(self._shares, dtype=np.uint32)
        prices = np.array(self._prices, dtype=np.uint64)
        new_order_ids = np.array(self._new_order_ids, dtype=np.uint64)

        (
            snap_ts,
            snap_bp,
            snap_bs,
            snap_ap,
            snap_as_,
            snap_spread,
            snap_mid,
        ) = _process_messages_numba(
            msg_types,
            timestamps,
            order_ids,
            sides,
            shares,
            prices,
            new_order_ids,
            uint64(self.snapshot_interval_us),
            self.max_snapshots,
        )

        self.snapshots = []
        for i in range(len(snap_ts)):
            self.snapshots.append(
                OrderBookSnapshot(
                    timestamp=snap_ts[i],
                    bid_prices=snap_bp[i].copy(),
                    bid_sizes=snap_bs[i].copy(),
                    ask_prices=snap_ap[i].copy(),
                    ask_sizes=snap_as_[i].copy(),
                    spread=float(snap_spread[i]),
                    midprice=float(snap_mid[i]),
                )
            )

    def to_polars(self) -> "pl.DataFrame":
        """
        将所有快照转换为 Polars DataFrame

        Returns
        -------
        pl.DataFrame
            包含时间戳、十档买卖盘价格/数量、价差、中间价的宽表
        """
        if pl is None:
            raise ImportError("polars 未安装")

        if not self.snapshots:
            return pl.DataFrame()

        rows = []
        for snap in self.snapshots:
            row = {
                "timestamp": int(snap.timestamp),
                "spread": snap.spread,
                "midprice": snap.midprice,
                "best_bid": snap.best_bid,
                "best_ask": snap.best_ask,
            }
            for i in range(DEPTH_LEVELS):
                row[f"bid_price_{i+1}"] = float(snap.bid_prices[i]) / 10000.0 if snap.bid_prices[i] > 0 else None
                row[f"bid_size_{i+1}"] = int(snap.bid_sizes[i]) if snap.bid_sizes[i] > 0 else None
                row[f"ask_price_{i+1}"] = float(snap.ask_prices[i]) / 10000.0 if snap.ask_prices[i] > 0 else None
                row[f"ask_size_{i+1}"] = int(snap.ask_sizes[i]) if snap.ask_sizes[i] > 0 else None
            rows.append(row)

        return pl.DataFrame(rows).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")).alias("datetime")
        )

    def spread_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取价差时间序列

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (时间戳数组, 价差数组)
        """
        if not self.snapshots:
            return np.array([], dtype=np.uint64), np.array([], dtype=np.float64)

        ts = np.array([int(s.timestamp) for s in self.snapshots], dtype=np.uint64)
        spread = np.array([s.spread for s in self.snapshots], dtype=np.float64)
        return ts, spread

    def midprice_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取中间价时间序列

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (时间戳数组, 中间价数组)
        """
        if not self.snapshots:
            return np.array([], dtype=np.uint64), np.array([], dtype=np.float64)

        ts = np.array([int(s.timestamp) for s in self.snapshots], dtype=np.uint64)
        mid = np.array([s.midprice for s in self.snapshots], dtype=np.float64)
        return ts, mid
