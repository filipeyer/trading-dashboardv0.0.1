#!/usr/bin/env python3
import ccxt
import psycopg2
import os
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import sys
import time

# ============================================================================
# DATABASE CONNECTION
# Reads from environment variable on GitHub Actions
# Falls back to hardcoded string on your laptop
# ============================================================================

CONNECTION_STRING = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres.ffpspjiznmupxassxxxs:ZjjebPPo4b2U9ci@aws-1-eu-west-1.pooler.supabase.com:5432/postgres'
)

# ============================================================================
# EXCHANGE CONFIGURATION
# ============================================================================

EXCHANGES = {
    'binance': {
        'enabled': True,
        'ccxt_class': ccxt.binance,
        'options': {},
        'pairs': ['BTC/USDT', 'ETH/USDT']
    },
    'bybit': {
        'enabled': True,
        'ccxt_class': ccxt.bybit,
        'options': {'defaultType': 'swap'},
        'pairs': ['BTC/USDT:USDT']
    },
    'coinbase': {
        'enabled': True,
        'ccxt_class': ccxt.coinbase,
        'options': {},
        'pairs': ['BTC/USD']
    },
}

TIMEFRAME = '15m'
INITIAL_LOOKBACK_DAYS = 30

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def connect_to_database():
    try:
        conn = psycopg2.connect(CONNECTION_STRING)
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

def get_last_timestamp(conn, exchange, symbol, timeframe):
    query = """
    SELECT MAX(timestamp)
    FROM ohlcv_data
    WHERE exchange = %s AND symbol = %s AND timeframe = %s
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (exchange, symbol, timeframe))
            result = cur.fetchone()[0]
        return result
    except:
        return None

def insert_candles(conn, candles_data):
    if not candles_data:
        return 0

    # Deduplicate
    seen = set()
    unique = []
    for r in candles_data:
        key = (r[0], r[1], r[2], r[3])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    insert_query = """
    INSERT INTO ohlcv_data
        (exchange, symbol, timeframe, timestamp, open, high, low, close, volume)
    VALUES %s
    ON CONFLICT (exchange, symbol, timeframe, timestamp)
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume
    """
    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_query, unique)
        conn.commit()
        return len(unique)
    except Exception as e:
        print(f"    ❌ Insert failed: {e}")
        conn.rollback()
        return 0

# ============================================================================
# FETCH FUNCTIONS
# ============================================================================

def fetch_candles_from_exchange(exchange_obj, symbol, timeframe, since_ms):
    try:
        if not exchange_obj.markets:
            exchange_obj.load_markets()
        if symbol not in exchange_obj.markets:
            return None, f"Symbol {symbol} not available"
        ohlcv = exchange_obj.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        return ohlcv, None
    except Exception as e:
        return None, str(e)

def process_candles(exchange_name, symbol, timeframe, raw_ohlcv):
    processed = []
    for candle in raw_ohlcv:
        ts = candle[0]
        if ts < 1e12:
            ts = ts * 1000
        dt = datetime.fromtimestamp(ts / 1000)
        processed.append((
            exchange_name, symbol, timeframe, dt,
            float(candle[1]), float(candle[2]),
            float(candle[3]), float(candle[4]),
            float(candle[5]) if candle[5] else 0.0
        ))
    return processed

# ============================================================================
# MAIN
# ============================================================================

def fetch_all_data():
    print("=" * 60)
    print("🚀 DATA FETCHER")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n📊 Connecting to database...")
    conn = connect_to_database()
    print("✅ Connected!")

    total_candles = 0
    total_pairs = 0

    for exchange_name, config in EXCHANGES.items():
        if not config['enabled']:
            continue

        print(f"\n📡 {exchange_name.upper()}")

        try:
            cfg = {'enableRateLimit': True}
            if config['options']:
                cfg['options'] = config['options']
            exchange_obj = config['ccxt_class'](cfg)
            exchange_obj.load_markets()
        except Exception as e:
            print(f"  ❌ Failed to init: {e}")
            continue

        for symbol in config['pairs']:
            print(f"  📊 {symbol}...", end=" ")
            total_pairs += 1

            try:
                last_ts = get_last_timestamp(conn, exchange_name, symbol, TIMEFRAME)

                if last_ts:
                    since_ms = int(last_ts.timestamp() * 1000)
                    print(f"(from {last_ts.strftime('%m/%d %H:%M')})", end=" ")
                else:
                    since_dt = datetime.now() - timedelta(days=INITIAL_LOOKBACK_DAYS)
                    since_ms = int(since_dt.timestamp() * 1000)
                    print(f"(initial {INITIAL_LOOKBACK_DAYS}d)", end=" ")

                raw_candles, error = fetch_candles_from_exchange(
                    exchange_obj, symbol, TIMEFRAME, since_ms
                )

                if error:
                    print(f"❌ {error}")
                    continue

                if not raw_candles:
                    print("⚠️  No new data")
                    continue

                processed = process_candles(exchange_name, symbol, TIMEFRAME, raw_candles)
                inserted = insert_candles(conn, processed)
                total_candles += inserted
                print(f"✅ {inserted} candles")

                time.sleep(exchange_obj.rateLimit / 1000)

            except Exception as e:
                print(f"❌ {str(e)[:50]}")
                continue

    conn.close()

    print("\n" + "=" * 60)
    print(f"✅ Pairs: {total_pairs} | Candles: {total_candles}")
    print(f"⏰ Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        fetch_all_data()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Stopped")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        sys.exit(1)
