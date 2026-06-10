"""
ITCH 5.0 协议高速解析器
使用 struct 按位解析纳斯达克 ITCH 5.0 协议报文
支持从 PCAP 网络抓包文件中提取逐笔订单消息
"""

import struct
import gzip
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator, List, Optional, Tuple

import numpy as np
import polars as pl


class ITCHMessageType(IntEnum):
    ADD_ORDER = ord("A")
    ADD_ORDER_MPID = ord("F")
    ORDER_EXECUTED = ord("E")
    ORDER_EXECUTED_WITH_PRICE = ord("C")
    ORDER_CANCEL = ord("X")
    ORDER_DELETE = ord("D")
    ORDER_REPLACE = ord("U")
    TRADE_NON_CROSS = ord("P")
    TRADE_CROSS = ord("Q")
    BROKEN_TRADE = ord("B")
    NOII = ord("I")
    RPII = ord("N")
    SYSTEM_EVENT = ord("S")
    STOCK_DIRECTORY = ord("R")
    STOCK_TRADING_ACTION = ord("H")
    REG_SHO_RESTRICTION = ord("Y")
    MWCB_DECLINE_LEVEL = ord("V")
    MWCB_STATUS = ord("W")
    IPO_QUOTING = ord("K")
    LULD_AUCTION_COLLAR = ord("J")
    OPERATIONAL_HALT = ord("h")


class Side(IntEnum):
    BUY = ord("B")
    SELL = ord("S")


@dataclass
class AddOrderMsg:
    timestamp: np.uint64
    order_id: np.uint64
    side: np.uint8
    shares: np.uint32
    stock: str
    price: np.uint64
    mpid: str = ""


@dataclass
class OrderExecutedMsg:
    timestamp: np.uint64
    order_id: np.uint64
    executed_shares: np.uint32
    match_number: np.uint64
    printable: bool = True
    execution_price: Optional[np.uint64] = None


@dataclass
class OrderCancelMsg:
    timestamp: np.uint64
    order_id: np.uint64
    canceled_shares: np.uint32


@dataclass
class OrderDeleteMsg:
    timestamp: np.uint64
    order_id: np.uint64


@dataclass
class OrderReplaceMsg:
    timestamp: np.uint64
    original_order_id: np.uint64
    new_order_id: np.uint64
    shares: np.uint32
    price: np.uint64


@dataclass
class TradeMsg:
    timestamp: np.uint64
    order_id: np.uint64
    side: np.uint8
    shares: np.uint32
    stock: str
    price: np.uint64
    match_number: np.uint64
    cross_type: np.uint8 = 0


ITCH_MSG = AddOrderMsg | OrderExecutedMsg | OrderCancelMsg | OrderDeleteMsg | OrderReplaceMsg | TradeMsg


LENGTH_PREFIX_FMT = ">H"
LENGTH_PREFIX_SIZE = struct.calcsize(LENGTH_PREFIX_FMT)


ADD_ORDER_FMT = ">6sQcI8sQ"
ADD_ORDER_SIZE = struct.calcsize(ADD_ORDER_FMT)
ADD_ORDER_MPID_FMT = ">6sQcI8sQ4s"
ADD_ORDER_MPID_SIZE = struct.calcsize(ADD_ORDER_MPID_FMT)
ORDER_EXECUTED_FMT = ">6sQIQ"
ORDER_EXECUTED_SIZE = struct.calcsize(ORDER_EXECUTED_FMT)
ORDER_EXECUTED_PRICE_FMT = ">6sQIQcQ"
ORDER_EXECUTED_PRICE_SIZE = struct.calcsize(ORDER_EXECUTED_PRICE_FMT)
ORDER_CANCEL_FMT = ">6sQI"
ORDER_CANCEL_SIZE = struct.calcsize(ORDER_CANCEL_FMT)
ORDER_DELETE_FMT = ">6sQ"
ORDER_DELETE_SIZE = struct.calcsize(ORDER_DELETE_FMT)
ORDER_REPLACE_FMT = ">6sQQIQ"
ORDER_REPLACE_SIZE = struct.calcsize(ORDER_REPLACE_FMT)
TRADE_FMT = ">6sQcI8sQQ"
TRADE_SIZE = struct.calcsize(TRADE_FMT)
TRADE_CROSS_FMT = ">6sQcI8sQQB"
TRADE_CROSS_SIZE = struct.calcsize(TRADE_CROSS_FMT)


def _decode_stock(b: bytes) -> str:
    return b.decode("ascii", errors="replace").strip()


def _decode_timestamp(b: bytes) -> np.uint64:
    return np.uint64(int.from_bytes(b, "big"))


def _itch_price_to_float(price_int: int) -> float:
    return float(price_int) / 10000.0


class ITCHParser:
    """
    ITCH 5.0 协议高速解析器

    支持从原始二进制 ITCH 数据或 PCAP 网络抓包文件中提取逐笔订单消息。
    利用 struct 模块按位解析，避免 Python 层循环开销。
    """

    def __init__(self, stock_filter: Optional[List[str]] = None):
        self.stock_filter = (
            {s.encode("ascii").ljust(8) for s in stock_filter}
            if stock_filter
            else None
        )
        self._reset_buffers()

    def _reset_buffers(self):
        self._add_orders: List[Tuple] = []
        self._order_executes: List[Tuple] = []
        self._order_cancels: List[Tuple] = []
        self._order_deletes: List[Tuple] = []
        self._order_replaces: List[Tuple] = []
        self._trades: List[Tuple] = []

    def parse_itch_file(self, filepath: str, max_messages: Optional[int] = None) -> None:
        """
        解析原始 ITCH 二进制文件（.itch / .itch.gz）

        Parameters
        ----------
        filepath : str
            ITCH 文件路径，支持 .gz 压缩
        max_messages : Optional[int]
            最大解析消息数（用于调试）
        """
        self._reset_buffers()

        if filepath.endswith(".gz"):
            opener = gzip.open
            mode = "rb"
        else:
            opener = open
            mode = "rb"

        count = 0
        with opener(filepath, mode) as f:
            while True:
                if max_messages and count >= max_messages:
                    break
                len_bytes = f.read(LENGTH_PREFIX_SIZE)
                if len(len_bytes) < LENGTH_PREFIX_SIZE:
                    break
                msg_len = struct.unpack(LENGTH_PREFIX_FMT, len_bytes)[0]
                payload = f.read(msg_len)
                if len(payload) < msg_len:
                    break
                self._parse_payload(payload)
                count += 1

    def parse_pcap_file(self, filepath: str, max_messages: Optional[int] = None) -> None:
        """
        解析 PCAP 网络抓包文件，提取 ITCH 5.0 净荷

        Parameters
        ----------
        filepath : str
            PCAP 文件路径
        max_messages : Optional[int]
            最大解析消息数
        """
        try:
            import dpkt

            self._reset_buffers()
            count = 0

            with open(filepath, "rb") as f:
                pcap = dpkt.pcap.Reader(f)
                for _, buf in pcap:
                    if max_messages and count >= max_messages:
                        break
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    if not isinstance(ip.data, dpkt.tcp.TCP):
                        continue
                    tcp = ip.data
                    payload = bytes(tcp.data)
                    if not payload:
                        continue
                    parsed = self._parse_tcp_payload(payload)
                    count += parsed
            return
        except ImportError:
            pass

        try:
            from scapy.all import rdpcap, TCP, IP

            self._reset_buffers()
            count = 0
            packets = rdpcap(filepath)
            for pkt in packets:
                if max_messages and count >= max_messages:
                    break
                if TCP in pkt and pkt[TCP].payload:
                    payload = bytes(pkt[TCP].payload)
                    if payload:
                        parsed = self._parse_tcp_payload(payload)
                        count += parsed
            return
        except ImportError:
            raise ImportError("需要安装 dpkt 或 scapy 来解析 PCAP 文件")

    def _parse_tcp_payload(self, payload: bytes) -> int:
        """
        从 TCP 净荷流中切割并解析 ITCH 消息
        返回解析的消息数
        """
        offset = 0
        n = len(payload)
        parsed = 0

        while offset + LENGTH_PREFIX_SIZE <= n:
            msg_len = struct.unpack_from(LENGTH_PREFIX_FMT, payload, offset)[0]
            if msg_len < 1 or offset + LENGTH_PREFIX_SIZE + msg_len > n:
                break
            msg_start = offset + LENGTH_PREFIX_SIZE
            msg_end = msg_start + msg_len
            self._parse_payload(payload[msg_start:msg_end])
            offset = msg_end
            parsed += 1

        return parsed

    def _parse_payload(self, payload: bytes) -> None:
        """解析单条 ITCH 消息净荷"""
        if not payload:
            return

        msg_type = payload[0]
        body = payload[1:]

        try:
            if msg_type == ITCHMessageType.ADD_ORDER:
                self._parse_add_order(body, has_mpid=False)
            elif msg_type == ITCHMessageType.ADD_ORDER_MPID:
                self._parse_add_order(body, has_mpid=True)
            elif msg_type == ITCHMessageType.ORDER_EXECUTED:
                self._parse_order_executed(body, has_price=False)
            elif msg_type == ITCHMessageType.ORDER_EXECUTED_WITH_PRICE:
                self._parse_order_executed(body, has_price=True)
            elif msg_type == ITCHMessageType.ORDER_CANCEL:
                self._parse_order_cancel(body)
            elif msg_type == ITCHMessageType.ORDER_DELETE:
                self._parse_order_delete(body)
            elif msg_type == ITCHMessageType.ORDER_REPLACE:
                self._parse_order_replace(body)
            elif msg_type == ITCHMessageType.TRADE_NON_CROSS:
                self._parse_trade(body, cross=False)
            elif msg_type == ITCHMessageType.TRADE_CROSS:
                self._parse_trade(body, cross=True)
        except struct.error:
            pass

    def _parse_add_order(self, body: bytes, has_mpid: bool) -> None:
        if has_mpid:
            if len(body) < ADD_ORDER_MPID_SIZE:
                return
            ts_bytes, order_id, side, shares, stock, price, mpid = struct.unpack(
                ADD_ORDER_MPID_FMT, body[:ADD_ORDER_MPID_SIZE]
            )
            mpid_str = mpid.decode("ascii", errors="replace").strip()
        else:
            if len(body) < ADD_ORDER_SIZE:
                return
            ts_bytes, order_id, side, shares, stock, price = struct.unpack(
                ADD_ORDER_FMT, body[:ADD_ORDER_SIZE]
            )
            mpid_str = ""

        if self.stock_filter and stock not in self.stock_filter:
            return

        self._add_orders.append(
            (
                _decode_timestamp(ts_bytes),
                np.uint64(order_id),
                np.uint8(side[0]),
                np.uint32(shares),
                _decode_stock(stock),
                np.uint64(price),
                mpid_str,
            )
        )

    def _parse_order_executed(self, body: bytes, has_price: bool) -> None:
        if has_price:
            if len(body) < ORDER_EXECUTED_PRICE_SIZE:
                return
            ts_bytes, order_id, shares, match_num, printable, price = struct.unpack(
                ORDER_EXECUTED_PRICE_FMT, body[:ORDER_EXECUTED_PRICE_SIZE]
            )
            self._order_executes.append(
                (
                    _decode_timestamp(ts_bytes),
                    np.uint64(order_id),
                    np.uint32(shares),
                    np.uint64(match_num),
                    printable == b"Y",
                    np.uint64(price),
                )
            )
        else:
            if len(body) < ORDER_EXECUTED_SIZE:
                return
            ts_bytes, order_id, shares, match_num = struct.unpack(
                ORDER_EXECUTED_FMT, body[:ORDER_EXECUTED_SIZE]
            )
            self._order_executes.append(
                (
                    _decode_timestamp(ts_bytes),
                    np.uint64(order_id),
                    np.uint32(shares),
                    np.uint64(match_num),
                    True,
                    None,
                )
            )

    def _parse_order_cancel(self, body: bytes) -> None:
        if len(body) < ORDER_CANCEL_SIZE:
            return
        ts_bytes, order_id, shares = struct.unpack(
            ORDER_CANCEL_FMT, body[:ORDER_CANCEL_SIZE]
        )
        self._order_cancels.append(
            (
                _decode_timestamp(ts_bytes),
                np.uint64(order_id),
                np.uint32(shares),
            )
        )

    def _parse_order_delete(self, body: bytes) -> None:
        if len(body) < ORDER_DELETE_SIZE:
            return
        ts_bytes, order_id = struct.unpack(
            ORDER_DELETE_FMT, body[:ORDER_DELETE_SIZE]
        )
        self._order_deletes.append(
            (
                _decode_timestamp(ts_bytes),
                np.uint64(order_id),
            )
        )

    def _parse_order_replace(self, body: bytes) -> None:
        if len(body) < ORDER_REPLACE_SIZE:
            return
        ts_bytes, orig_id, new_id, shares, price = struct.unpack(
            ORDER_REPLACE_FMT, body[:ORDER_REPLACE_SIZE]
        )
        self._order_replaces.append(
            (
                _decode_timestamp(ts_bytes),
                np.uint64(orig_id),
                np.uint64(new_id),
                np.uint32(shares),
                np.uint64(price),
            )
        )

    def _parse_trade(self, body: bytes, cross: bool) -> None:
        if cross:
            if len(body) < TRADE_CROSS_SIZE:
                return
            ts_bytes, order_id, side, shares, stock, price, match_num, cross_type = struct.unpack(
                TRADE_CROSS_FMT, body[:TRADE_CROSS_SIZE]
            )
        else:
            if len(body) < TRADE_SIZE:
                return
            ts_bytes, order_id, side, shares, stock, price, match_num = struct.unpack(
                TRADE_FMT, body[:TRADE_SIZE]
            )
            cross_type = 0

        if self.stock_filter and stock not in self.stock_filter:
            return

        self._trades.append(
            (
                _decode_timestamp(ts_bytes),
                np.uint64(order_id),
                np.uint8(side[0]),
                np.uint32(shares),
                _decode_stock(stock),
                np.uint64(price),
                np.uint64(match_num),
                np.uint8(cross_type),
            )
        )

    def to_polars(self) -> dict[str, pl.DataFrame]:
        """
        将所有解析结果转换为 Polars DataFrame 字典

        Returns
        -------
        dict[str, pl.DataFrame]
            键为消息类型，值为对应的 Polars DataFrame
        """
        result = {}

        if self._add_orders:
            result["add_orders"] = pl.DataFrame(
                self._add_orders,
                schema={
                    "timestamp": pl.UInt64,
                    "order_id": pl.UInt64,
                    "side": pl.UInt8,
                    "shares": pl.UInt32,
                    "stock": pl.Utf8,
                    "price": pl.UInt64,
                    "mpid": pl.Utf8,
                },
                orient="row",
            ).with_columns(
                (pl.col("price") / 10000.0).alias("price_float"),
                pl.col("timestamp").cast(pl.Datetime("us")).alias("datetime"),
            )

        if self._order_executes:
            rows_ts = []
            rows_oid = []
            rows_shares = []
            rows_match = []
            rows_printable = []
            rows_exec_price = []
            for r in self._order_executes:
                rows_ts.append(int(r[0]))
                rows_oid.append(int(r[1]))
                rows_shares.append(int(r[2]))
                rows_match.append(int(r[3]))
                rows_printable.append(r[4])
                rows_exec_price.append(float(r[5]) / 10000.0 if r[5] is not None else None)
            result["order_executes"] = pl.DataFrame(
                {
                    "timestamp": rows_ts,
                    "order_id": rows_oid,
                    "shares": rows_shares,
                    "match_number": rows_match,
                    "printable": rows_printable,
                    "execution_price": rows_exec_price,
                },
                schema={
                    "timestamp": pl.UInt64,
                    "order_id": pl.UInt64,
                    "shares": pl.UInt32,
                    "match_number": pl.UInt64,
                    "printable": pl.Boolean,
                    "execution_price": pl.Float64,
                },
            ).with_columns(
                pl.col("timestamp").cast(pl.Datetime("us")).alias("datetime")
            )

        if self._order_cancels:
            result["order_cancels"] = pl.DataFrame(
                self._order_cancels,
                schema={
                    "timestamp": pl.UInt64,
                    "order_id": pl.UInt64,
                    "canceled_shares": pl.UInt32,
                },
                orient="row",
            ).with_columns(
                pl.col("timestamp").cast(pl.Datetime("us")).alias("datetime")
            )

        if self._order_deletes:
            result["order_deletes"] = pl.DataFrame(
                self._order_deletes,
                schema={
                    "timestamp": pl.UInt64,
                    "order_id": pl.UInt64,
                },
                orient="row",
            ).with_columns(
                pl.col("timestamp").cast(pl.Datetime("us")).alias("datetime")
            )

        if self._order_replaces:
            result["order_replaces"] = pl.DataFrame(
                self._order_replaces,
                schema={
                    "timestamp": pl.UInt64,
                    "original_order_id": pl.UInt64,
                    "new_order_id": pl.UInt64,
                    "shares": pl.UInt32,
                    "price": pl.UInt64,
                },
                orient="row",
            ).with_columns(
                (pl.col("price") / 10000.0).alias("price_float"),
                pl.col("timestamp").cast(pl.Datetime("us")).alias("datetime"),
            )

        if self._trades:
            result["trades"] = pl.DataFrame(
                self._trades,
                schema={
                    "timestamp": pl.UInt64,
                    "order_id": pl.UInt64,
                    "side": pl.UInt8,
                    "shares": pl.UInt32,
                    "stock": pl.Utf8,
                    "price": pl.UInt64,
                    "match_number": pl.UInt64,
                    "cross_type": pl.UInt8,
                },
                orient="row",
            ).with_columns(
                (pl.col("price") / 10000.0).alias("price_float"),
                pl.col("timestamp").cast(pl.Datetime("us")).alias("datetime"),
            )

        return result

    def generate_synthetic(
        self,
        stock: str = "TEST",
        n_messages: int = 100000,
        seed: int = 42,
    ) -> None:
        """
        生成合成 ITCH 数据用于测试

        Parameters
        ----------
        stock : str
            股票代码
        n_messages : int
            消息数量
        seed : int
            随机种子
        """
        self._reset_buffers()
        rng = np.random.default_rng(seed)

        stock_bytes = stock.encode("ascii").ljust(8)
        mid_price = np.uint64(1000000)
        best_bid = np.uint64(int(mid_price) - 5)
        best_ask = np.uint64(int(mid_price) + 5)
        order_id_counter = np.uint64(1)
        ts = np.uint64(0)

        active_buy_orders: dict[np.uint64, tuple[np.uint32, np.uint64]] = {}
        active_sell_orders: dict[np.uint64, tuple[np.uint32, np.uint64]] = {}

        def _update_best():
            nonlocal best_bid, best_ask
            if active_buy_orders:
                best_bid = max(p for _, p in active_buy_orders.values())
            if active_sell_orders:
                best_ask = min(p for _, p in active_sell_orders.values())

        for _ in range(n_messages):
            ts += rng.integers(1, 100, dtype=np.uint64)
            msg_choice = rng.random()

            if msg_choice < 0.4:
                side = Side.BUY if rng.random() < 0.5 else Side.SELL
                if side == Side.BUY:
                    max_bid = int(best_ask) - 1 if best_ask > best_bid + 1 else int(best_bid)
                    min_bid = max(1, int(best_bid) - 50)
                    if min_bid >= max_bid:
                        price = np.uint64(max_bid)
                    else:
                        price = np.uint64(rng.integers(min_bid, max_bid + 1, dtype=np.int64))
                else:
                    min_ask = int(best_bid) + 1 if best_ask > best_bid + 1 else int(best_ask)
                    max_ask = int(best_ask) + 50
                    if min_ask >= max_ask:
                        price = np.uint64(min_ask)
                    else:
                        price = np.uint64(rng.integers(min_ask, max_ask + 1, dtype=np.int64))
                shares = rng.integers(10, 1000, dtype=np.uint32)
                order_id = order_id_counter
                order_id_counter += np.uint64(1)

                self._add_orders.append(
                    (ts, order_id, np.uint8(side), shares, stock, price, "")
                )

                if side == Side.BUY:
                    active_buy_orders[order_id] = (shares, price)
                    if price > best_bid:
                        best_bid = price
                else:
                    active_sell_orders[order_id] = (shares, price)
                    if price < best_ask:
                        best_ask = price

            elif msg_choice < 0.6:
                is_buy_side = False
                if rng.random() < 0.5 and active_buy_orders:
                    order_id = rng.choice(list(active_buy_orders.keys()))
                    remaining_shares, price = active_buy_orders[order_id]
                    is_buy_side = True
                elif active_sell_orders:
                    order_id = rng.choice(list(active_sell_orders.keys()))
                    remaining_shares, price = active_sell_orders[order_id]
                    is_buy_side = False
                else:
                    continue

                if remaining_shares <= 0:
                    continue

                exec_shares = rng.integers(
                    1, max(2, int(remaining_shares) + 1), dtype=np.uint32
                )
                exec_shares = min(exec_shares, remaining_shares)

                self._order_executes.append(
                    (
                        ts,
                        order_id,
                        exec_shares,
                        order_id_counter,
                        True,
                        None,
                    )
                )

                trade_side = np.uint8(ord('S')) if is_buy_side else np.uint8(ord('B'))
                self._trades.append(
                    (
                        ts,
                        order_id,
                        trade_side,
                        exec_shares,
                        stock,
                        price,
                        order_id_counter,
                        np.uint8(0),
                    )
                )
                order_id_counter += np.uint64(1)

                new_remaining = remaining_shares - exec_shares
                if new_remaining <= 0:
                    removed_bid = order_id in active_buy_orders
                    removed_ask = order_id in active_sell_orders
                    active_buy_orders.pop(order_id, None)
                    active_sell_orders.pop(order_id, None)
                    if removed_bid or removed_ask:
                        _update_best()
                else:
                    if order_id in active_buy_orders:
                        active_buy_orders[order_id] = (np.uint32(new_remaining), price)
                    else:
                        active_sell_orders[order_id] = (np.uint32(new_remaining), price)

            elif msg_choice < 0.75:
                if rng.random() < 0.5 and active_buy_orders:
                    order_id = rng.choice(list(active_buy_orders.keys()))
                    remaining_shares, price = active_buy_orders[order_id]
                    side_buy = True
                elif active_sell_orders:
                    order_id = rng.choice(list(active_sell_orders.keys()))
                    remaining_shares, price = active_sell_orders[order_id]
                    side_buy = False
                else:
                    continue

                if remaining_shares <= 1:
                    continue

                cancel_shares = rng.integers(
                    1, int(remaining_shares), dtype=np.uint32
                )
                self._order_cancels.append(
                    (ts, order_id, cancel_shares)
                )

                new_remaining = remaining_shares - cancel_shares
                if side_buy:
                    active_buy_orders[order_id] = (np.uint32(new_remaining), price)
                else:
                    active_sell_orders[order_id] = (np.uint32(new_remaining), price)

            elif msg_choice < 0.85:
                removed = False
                if rng.random() < 0.5 and active_buy_orders:
                    order_id = rng.choice(list(active_buy_orders.keys()))
                    active_buy_orders.pop(order_id, None)
                    removed = True
                elif active_sell_orders:
                    order_id = rng.choice(list(active_sell_orders.keys()))
                    active_sell_orders.pop(order_id, None)
                    removed = True
                else:
                    continue

                if removed:
                    _update_best()
                self._order_deletes.append((ts, order_id))

            else:
                if rng.random() < 0.5 and active_buy_orders:
                    order_id = rng.choice(list(active_buy_orders.keys()))
                    old_shares, old_price = active_buy_orders[order_id]
                    side = Side.BUY
                    side_buy = True
                elif active_sell_orders:
                    order_id = rng.choice(list(active_sell_orders.keys()))
                    old_shares, old_price = active_sell_orders[order_id]
                    side = Side.SELL
                    side_buy = False
                else:
                    continue

                new_order_id = order_id_counter
                order_id_counter += np.uint64(1)
                new_shares = rng.integers(10, 1000, dtype=np.uint32)
                if side_buy:
                    max_bid = int(best_ask) - 1 if best_ask > best_bid + 1 else int(best_bid)
                    min_bid = max(1, int(best_bid) - 50)
                    if min_bid >= max_bid:
                        new_price = np.uint64(max_bid)
                    else:
                        new_price = np.uint64(rng.integers(min_bid, max_bid + 1, dtype=np.int64))
                else:
                    min_ask = int(best_bid) + 1 if best_ask > best_bid + 1 else int(best_ask)
                    max_ask = int(best_ask) + 50
                    if min_ask >= max_ask:
                        new_price = np.uint64(min_ask)
                    else:
                        new_price = np.uint64(rng.integers(min_ask, max_ask + 1, dtype=np.int64))

                self._order_replaces.append(
                    (ts, order_id, new_order_id, new_shares, new_price)
                )

                if side == Side.BUY:
                    active_buy_orders.pop(order_id, None)
                    active_buy_orders[new_order_id] = (new_shares, new_price)
                else:
                    active_sell_orders.pop(order_id, None)
                    active_sell_orders[new_order_id] = (new_shares, new_price)
                _update_best()

            if rng.random() < 0.01:
                delta = int(rng.integers(-5, 5, dtype=np.int64))
                if delta > 0:
                    best_ask = np.uint64(int(best_ask) + delta)
                    best_bid = np.uint64(int(best_bid) + delta)
                elif delta < 0:
                    best_ask = np.uint64(max(10, int(best_ask) + delta))
                    best_bid = np.uint64(max(10, int(best_bid) + delta))
                mid_price = np.uint64((int(best_bid) + int(best_ask)) // 2)
