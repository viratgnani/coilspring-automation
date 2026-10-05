import re
import os

filepath = 'src/coilbrain/bybit_demo.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Add instrument cache to BybitDemoClient
content = content.replace(
    '        self._timeout = timeout_s',
    '        self._timeout = timeout_s\n        self._instruments_cache: dict[str, dict] = {}'
)

new_methods = """
    async def get_instrument_info(self, symbol: str) -> dict:
        \"\"\"Fetch and cache instrument metadata for step/tick rounding.\"\"\"
        if symbol in self._instruments_cache:
            return self._instruments_cache[symbol]

        data = await self._get(
            "/v5/market/instruments-info",
            params={"category": "linear", "symbol": symbol},
            signed=False,
        )
        instruments = data.get("result", {}).get("list", [])
        for inst in instruments:
            if inst.get("symbol") == symbol and inst.get("status") == "Trading":
                self._instruments_cache[symbol] = inst
                return inst
        raise BybitDemoConfigError(f"Instrument {symbol} not found or not trading.")

    def round_qty(self, qty: float, info: dict) -> str:
        \"\"\"Round quantity to the step size defined by lotSizeFilter.\"\"\"
        lot_filter = info.get("lotSizeFilter", {})
        step = float(lot_filter.get("qtyStep", "0.001"))
        qty_rounded = int(qty / step) * step
        # Format without scientific notation, removing trailing zeros
        return f"{qty_rounded:.10f}".rstrip("0").rstrip(".")

    def round_price(self, price: float, info: dict) -> str:
        \"\"\"Round price to the tick size defined by priceFilter.\"\"\"
        tick = float(info.get("priceFilter", {}).get("tickSize", "0.1"))
        price_rounded = round(price / tick) * tick
        return f"{price_rounded:.10f}".rstrip("0").rstrip(".")

    async def validate_symbol(self, symbol: str) -> bool:
"""

content = re.sub(r'    async def validate_symbol\(self, symbol: str\) -> bool:', new_methods, content, count=1)

content = content.replace(
    '            "qty": str(qty),',
    '            "qty": self.round_qty(qty, await self.get_instrument_info(symbol)),'
)

content = content.replace(
    '            "stopLoss": str(sl_price),',
    '            "stopLoss": self.round_price(sl_price, await self.get_instrument_info(symbol)),'
)

content = content.replace(
    '            "takeProfit": str(tp_price),',
    '            "takeProfit": self.round_price(tp_price, await self.get_instrument_info(symbol)),'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched bybit_demo.py")
