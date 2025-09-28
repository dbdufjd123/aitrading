import ccxt, os
from dotenv import load_dotenv
load_dotenv()

class ExchangeWrapper:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        key = os.getenv("BINANCE_KEY", "")
        secret = os.getenv("BINANCE_SECRET", "")
        self.ex = ccxt.binance({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

    def fetch_orderbook(self, symbol="BTC/USDT", limit=10):
        return self.ex.fetch_order_book(symbol, limit=limit)

    def create_market_order(self, symbol, side, amount):
        if self.dry_run:
            return {"info":"DRY_RUN","symbol":symbol,"side":side,"amount":amount}
        return self.ex.create_order(symbol, type="market", side=side, amount=amount)

    def fetch_price(self, symbol="BTC/USDT"):
        return self.ex.fetch_ticker(symbol)["last"]
