#!/usr/bin/env python3
import ccxt
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import sys
import time

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

CONNECTION_STRING = "postgresql://postgres.ffpspjiznmupxassxxxs:ZjjebPPo4b2U9ci@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# ============================================================================
# EXCHANGE CONFIGURATION
# ============================================================================

EXCHANGES = {
    'binance': {
        'enabled': True,
        'ccxt_class': ccxt.binance,
        'options': {},
        'pairs': [
            'BTC/USDT',
            'ETH/USDT',
        ]
    },
    'bybit': {
        'enabled': True,
        'ccxt_class': ccxt.bybit,
        'options': {'defaultType': 'swap'},
        'pairs': [
            'BTC/USDT:USDT',  # BTC perpetual
        ]
    },
    'coinbase': {
        'enabled': True,
        'ccxt_class': ccxt.coinbase,
        'options': {},
        'pairs': [
            'BTC/USD',
        ]
    },
}

TIMEFRAME = '15m'
INITIAL_LOOKBACK_DAYS = 365  # Get 1 full year of history!

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
            execute_values(cur, insert_query, candles_data)
        conn.commit()
        return len(candles_data)
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        conn.rollback()
        return 0

# ============================================================================
# EXCHANGE FUNCTIONS - NOW WITH LOOP FOR FULL HISTORY!
# ============================================================================

def fetch_all_candles(exchange_obj, symbol, timeframe, since_ms):
    """
    Keeps fetching 1000 candles at a time until we have everything!
    This is how we get full history instead of just 1000 candles.
    """
    all_candles = []
    current_since = since_ms
    
    while True:
        try:
            candles = exchange_obj.fetch_ohlcv(
                symbol, 
                timeframe, 
                since=current_since, 
                limit=1000
            )
            
            # No more candles? We're done!
            if not candles or len(candles) == 0:
                break
            
            all_candles.extend(candles)
            
            # If we got less than 1000, we've reached the end
            if len(candles) < 1000:
                break
            
            # Move forward to fetch next batch
            current_since = candles[-1][0] + 1
            
            # Show progress
            last_dt = datetime.fromtimestamp(candles[-1][0] / 1000)
            print(f"\n    📦 Got {len(all_candles)} candles so far... (up to {last_dt.strftime('%Y-%m-%d')})", end="")
            
            # Small delay to respect rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n    ⚠️  Error fetching batch: {e}")
            break
    
    return all_candles

def process_candles(exchange_name, symbol, timeframe, raw_ohlcv):
    processed = []
    for candle in raw_ohlcv:
        timestamp_ms, open_, high, low, close, volume = candle
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        processed.append((
            exchange_name, symbol, timeframe, dt,
            float(open_), float(high), float(low), float(close),
            float(volume) if volume else 0.0
        ))
    return processed

# ============================================================================
# MAIN FETCH LOGIC
# ============================================================================

def fetch_all_data():
    print("=" * 60)
    print("🚀 BTC/ETH DATA FETCHER")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n📊 Connecting to database...")
    conn = connect_to_database()
    print("✅ Connected!")

    total_candles = 0
    total_pairs = 0

    for exchange_name, config in EXCHANGES.items():
        if not config['enabled']:
            continue

        print(f"\n📡 FETCHING FROM {exchange_name.upper()}")

        try:
            options = {'enableRateLimit': True}
            options.update(config['options'])
            exchange_obj = config['ccxt_class'](options)
            exchange_obj.load_markets()
        except Exception as e:
            print(f"❌ Failed to initialize {exchange_name}: {e}")
            continue

        for symbol in config['pairs']:
            print(f"\n  📊 {symbol}...", end=" ")
            total_pairs += 1

            try:
                # Check if symbol exists on this exchange
                if symbol not in exchange_obj.markets:
                    print(f"❌ Not available on {exchange_name}")
                    continue

                last_ts = get_last_timestamp(conn, exchange_name, symbol, TIMEFRAME)

                if last_ts:
                    # Update mode - just get new candles
                    since_ms = int(last_ts.timestamp() * 1000)
                    print(f"(update from {last_ts.strftime('%Y-%m-%d')})", end="")
                else:
                    # First time - get full history!
                    since_dt = datetime.now() - timedelta(days=INITIAL_LOOKBACK_DAYS)
                    since_ms = int(since_dt.timestamp() * 1000)
                    print(f"(full history - {INITIAL_LOOKBACK_DAYS} days)", end="")

                # Fetch ALL candles with loop
                raw_candles = fetch_all_candles(
                    exchange_obj, symbol, TIMEFRAME, since_ms
                )

                if not raw_candles:
                    print(f"\n  ⚠️  No data found")
                    continue

                # Process and insert
                processed = process_candles(exchange_name, symbol, TIMEFRAME, raw_candles)
                inserted = insert_candles(conn, processed)
                total_candles += inserted
                print(f"\n  ✅ {inserted} candles inserted!")

                # Delay between symbols
                time.sleep(1)

            except Exception as e:
                print(f"\n  ❌ Error: {str(e)[:80]}")
                continue

    conn.close()

    print("\n" + "=" * 60)
    print(f"✅ Pairs processed: {total_pairs}")
    print(f"✅ Total candles inserted: {total_candles}")
    print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return total_candles

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n🎯 BTC/ETH Data Fetcher - Ready to rock!\n")
    try:
        candles = fetch_all_data()
        print("\n✅ SUCCESS! Check your Supabase table now!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
