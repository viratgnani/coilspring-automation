import asyncio
import httpx
import json

async def get_inst():
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api-demo.bybit.com/v5/market/instruments-info?category=linear&symbol=BTCUSDT")
        print(json.dumps(resp.json(), indent=2))

asyncio.run(get_inst())
