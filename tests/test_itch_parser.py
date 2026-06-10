"""
单元测试 - ITCH 5.0 协议解析器
"""

import sys
import os
import tempfile
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hft_lob.itch_parser import (
    ITCHParser,
    ITCHMessageType,
    Side,
    ADD_ORDER_FMT,
    ADD_ORDER_SIZE,
    ORDER_EXECUTED_FMT,
    ORDER_EXECUTED_SIZE,
    ORDER_CANCEL_FMT,
    ORDER_CANCEL_SIZE,
    ORDER_DELETE_FMT,
    ORDER_DELETE_SIZE,
    ORDER_REPLACE_FMT,
    ORDER_REPLACE_SIZE,
    LENGTH_PREFIX_FMT,
    LENGTH_PREFIX_SIZE,
)


def _build_itch_msg(msg_type_byte: bytes, body: bytes) -> bytes:
    full_msg = msg_type_byte + body
    msg_len = len(full_msg)
    return struct.pack(LENGTH_PREFIX_FMT, msg_len) + full_msg


def test_add_order_parsing():
    parser = ITCHParser()

    ts = (1000).to_bytes(6, "big")
    order_id = 1
    side = b"B"
    shares = 100
    stock = b"AAPL    "
    price = 1500000

    body = struct.pack(ADD_ORDER_FMT, ts, order_id, side, shares, stock, price)
    raw = _build_itch_msg(b"A", body)

    parser._parse_payload(raw[LENGTH_PREFIX_SIZE:])

    result = parser.to_polars()
    assert "add_orders" in result
    df = result["add_orders"]
    assert len(df) == 1
    assert int(df["order_id"][0]) == 1
    assert int(df["side"][0]) == Side.BUY
    assert int(df["shares"][0]) == 100
    assert df["stock"][0].strip() == "AAPL"
    assert float(df["price_float"][0]) == 150.0
    print("[PASS] test_add_order_parsing")


def test_order_executed_parsing():
    parser = ITCHParser()

    ts = (2000).to_bytes(6, "big")
    order_id = 1
    shares = 50
    match_num = 999

    body = struct.pack(ORDER_EXECUTED_FMT, ts, order_id, shares, match_num)
    raw = _build_itch_msg(b"E", body)

    parser._parse_payload(raw[LENGTH_PREFIX_SIZE:])

    result = parser.to_polars()
    assert "order_executes" in result
    df = result["order_executes"]
    assert len(df) == 1
    assert int(df["order_id"][0]) == 1
    assert int(df["shares"][0]) == 50
    print("[PASS] test_order_executed_parsing")


def test_order_cancel_parsing():
    parser = ITCHParser()

    ts = (3000).to_bytes(6, "big")
    order_id = 1
    shares = 20

    body = struct.pack(ORDER_CANCEL_FMT, ts, order_id, shares)
    raw = _build_itch_msg(b"X", body)

    parser._parse_payload(raw[LENGTH_PREFIX_SIZE:])

    result = parser.to_polars()
    assert "order_cancels" in result
    df = result["order_cancels"]
    assert len(df) == 1
    assert int(df["order_id"][0]) == 1
    assert int(df["canceled_shares"][0]) == 20
    print("[PASS] test_order_cancel_parsing")


def test_order_delete_parsing():
    parser = ITCHParser()

    ts = (4000).to_bytes(6, "big")
    order_id = 1

    body = struct.pack(ORDER_DELETE_FMT, ts, order_id)
    raw = _build_itch_msg(b"D", body)

    parser._parse_payload(raw[LENGTH_PREFIX_SIZE:])

    result = parser.to_polars()
    assert "order_deletes" in result
    df = result["order_deletes"]
    assert len(df) == 1
    assert int(df["order_id"][0]) == 1
    print("[PASS] test_order_delete_parsing")


def test_order_replace_parsing():
    parser = ITCHParser()

    ts = (5000).to_bytes(6, "big")
    orig_id = 1
    new_id = 2
    shares = 500
    price = 2000000

    body = struct.pack(ORDER_REPLACE_FMT, ts, orig_id, new_id, shares, price)
    raw = _build_itch_msg(b"U", body)

    parser._parse_payload(raw[LENGTH_PREFIX_SIZE:])

    result = parser.to_polars()
    assert "order_replaces" in result
    df = result["order_replaces"]
    assert len(df) == 1
    assert int(df["original_order_id"][0]) == 1
    assert int(df["new_order_id"][0]) == 2
    assert float(df["price_float"][0]) == 200.0
    print("[PASS] test_order_replace_parsing")


def test_stock_filter():
    parser = ITCHParser(stock_filter=["AAPL"])

    ts = (1000).to_bytes(6, "big")
    for stock_name in [b"AAPL    ", b"MSFT    "]:
        body = struct.pack(ADD_ORDER_FMT, ts, 1, b"B", 100, stock_name, 1500000)
        raw = _build_itch_msg(b"A", body)
        parser._parse_payload(raw[LENGTH_PREFIX_SIZE:])

    result = parser.to_polars()
    assert len(result["add_orders"]) == 1
    assert result["add_orders"]["stock"][0].strip() == "AAPL"
    print("[PASS] test_stock_filter")


def test_synthetic_data_generation():
    parser = ITCHParser()
    parser.generate_synthetic(stock="TEST", n_messages=10000, seed=42)
    result = parser.to_polars()

    assert "add_orders" in result
    assert len(result["add_orders"]) > 0
    assert "order_executes" in result
    assert "order_cancels" in result
    assert "order_deletes" in result
    assert "order_replaces" in result

    total_msgs = sum(len(df) for df in result.values())
    assert total_msgs > 0
    print(f"[PASS] test_synthetic_data_generation (总消息数: {total_msgs:,})")


if __name__ == "__main__":
    test_add_order_parsing()
    test_order_executed_parsing()
    test_order_cancel_parsing()
    test_order_delete_parsing()
    test_order_replace_parsing()
    test_stock_filter()
    test_synthetic_data_generation()
    print("\n所有 ITCH Parser 测试通过!")
