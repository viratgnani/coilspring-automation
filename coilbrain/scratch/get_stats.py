import asyncio, aiosqlite
from coilbrain.research.analysis import regime_core

async def main():
    async with aiosqlite.connect('research_snapshot.db') as db:
        res = await regime_core.build_regime_dataset(db, min_sample=30)
        s = res['dataset_summary']
        print(f"TOTAL_SIGNALS: {s['TOTAL_SIGNALS']}")
        print(f"T0_BASED: {s['T0_BASED_COUNTS']}")
        print(f"PREVIOUS_HORIZON_FALLBACK: {s['PREVIOUS_HORIZON_FALLBACK_COUNTS']}")
        print(f"HYBRID: {s['HYBRID_COUNTS']}")
        print(f"INSUFFICIENT_DATA: {s['INSUFFICIENT_DATA_COUNTS']}")
        
        # GUN state timeline
        gun_states = [state for state in res['states'] if state['symbol'] == 'GUN/USDT:USDT' or 'GUN' in state['symbol']]
        for state in gun_states:
             print(f"GUN at {state['information_horizon']}s: State={state['state']}, Basis={state['regime_information_basis']}")
             
asyncio.run(main())
