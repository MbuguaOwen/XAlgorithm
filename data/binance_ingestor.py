# data/binance_ingestor.py – Resilient Binance Stream Ingestor with Async Callback Support

import asyncio
import json
import websockets
import logging
import socket
from datetime import datetime, timezone

class BinanceIngestor:
    def __init__(self, ws_url: str = None):
        """Initialize the ingestor with an optional custom WebSocket URL."""
        self.ws_url = ws_url or (
            "wss://stream.binance.com:9443/stream?streams="
            "btcusdt@ticker/ethusdt@ticker/ethbtc@ticker"
        )
        self.latest = {}

        # Normalize rare stream name issues
        self.symbol_map = {
            "BTCCUSDT": "BTCUSDT",
            "ETHHUSDT": "ETHUSDT",
        }

    async def stream(self, callback):
        retry_delay = 3
        max_delay = 60

        while True:
            try:
                logging.info("🔌 [Binance] Connecting to WebSocket...")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    logging.info("✅ [Binance] Connected. Listening for tick data...")
                    retry_delay = 3  # Reset after success

                    async for msg in websocket:
                        try:
                            data = json.loads(msg)
                            stream = data.get("stream", "")
                            symbol = stream.split("@")[0].upper()
                            symbol = self.symbol_map.get(symbol, symbol)

                            if symbol not in ["BTCUSDT", "ETHUSDT", "ETHBTC"]:
                                continue

                            payload = data["data"]
                            price = float(payload["c"])
                            ts = datetime.fromtimestamp(payload["E"] / 1000, tz=timezone.utc)

                            self.latest[symbol] = {
                                "price": price,
                                "timestamp": ts
                            }

                            if all(k in self.latest for k in ["BTCUSDT", "ETHUSDT", "ETHBTC"]):
                                btc_price = self.latest["BTCUSDT"]["price"]
                                eth_price = self.latest["ETHUSDT"]["price"]
                                ethbtc_price = self.latest["ETHBTC"]["price"]
                                callback_time = ts

                                # Debug check
                                logging.debug(f"[BinanceIngestor] callback_time type: {type(callback_time)}")

                                try:
                                    result = callback(callback_time, btc_price, eth_price, ethbtc_price)
                                    if asyncio.iscoroutine(result):
                                        await result
                                except Exception as cb_err:
                                    logging.warning(f"⚠️ Callback error: {cb_err}")

                        except Exception as parse_err:
                            logging.warning(f"⚠️ Error parsing message: {parse_err} | Raw: {msg}")

            except websockets.exceptions.ConnectionClosedError as e:
                logging.error(f"🔌 Connection closed: {e} — Retrying in {retry_delay}s")
            except socket.gaierror as e:
                logging.error(f"🌍 DNS resolution failed: {e} — Check VPN/network")
            except ConnectionRefusedError as e:
                logging.error(f"🚫 Connection refused: {e}")
            except Exception as e:
                logging.error(f"❌ Unexpected error: {e}")

            logging.info(f"🔁 Reconnecting in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)


# ─────────────────────────────────────────────
# 🧪 Standalone Test Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestor = BinanceIngestor()

    async def print_callback(ts, btc, eth, ethbtc):
        print(f"[{ts.isoformat()}] BTC: {btc:.2f} | ETH: {eth:.2f} | ETHBTC: {ethbtc:.6f}")

    asyncio.run(ingestor.stream(print_callback))
