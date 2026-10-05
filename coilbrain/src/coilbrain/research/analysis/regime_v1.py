"""
Regime Classification V1 for Research Analysis Engine.
"""

REGIME_DEFINITION_VERSION = "v1"

from typing import Dict, Any, Union

def classify_state(
    price_return_from_t0: float | None,
    oi_delta_from_t0_pct: float | None,
    trade_imbalance: float | None,
    price_change_prev: float | None = None,
    oi_change_prev: float | None = None,
    trade_imbalance_delta: float | None = None
) -> Dict[str, Any]:
    """
    Classify the market state based purely on point-in-time features,
    with explicit fallback to point-to-point previous horizon deltas if T0 is missing.
    """
    
    price_basis = "MISSING"
    oi_basis = "MISSING"
    trade_basis = "MISSING"
    
    effective_price = None
    effective_oi = None
    effective_trade = None
    
    # 1. Price
    if price_return_from_t0 is not None:
        effective_price = price_return_from_t0
        price_basis = "T0"
    elif price_change_prev is not None:
        effective_price = price_change_prev
        price_basis = "PREVIOUS_HORIZON"
        
    # 2. OI
    if oi_delta_from_t0_pct is not None:
        effective_oi = oi_delta_from_t0_pct
        oi_basis = "T0"
    elif oi_change_prev is not None:
        effective_oi = oi_change_prev
        oi_basis = "PREVIOUS_HORIZON"
        
    # 3. Trade
    if trade_imbalance is not None:
        effective_trade = trade_imbalance
        trade_basis = "CURRENT"
    elif trade_imbalance_delta is not None:
        effective_trade = trade_imbalance_delta
        trade_basis = "DELTA_PREVIOUS_HORIZON"
        
    # Determine information basis
    if effective_price is None or effective_oi is None or effective_trade is None:
        info_basis = "INSUFFICIENT_DATA"
    elif price_basis == "T0" and oi_basis == "T0" and trade_basis == "CURRENT":
        info_basis = "T0_BASED"
    elif price_basis == "PREVIOUS_HORIZON" and oi_basis == "PREVIOUS_HORIZON" and trade_basis == "DELTA_PREVIOUS_HORIZON":
        info_basis = "PREVIOUS_HORIZON_FALLBACK"
    else:
        info_basis = "HYBRID"

    state = "INSUFFICIENT_DATA"
    
    if info_basis != "INSUFFICIENT_DATA":
        price_up = effective_price > 0
        price_down = effective_price < 0
        price_flat = effective_price == 0
        
        oi_up = effective_oi > 0
        oi_down = effective_oi < 0
        oi_flat = effective_oi == 0
        
        trade_imb_up = effective_trade > 0
        trade_imb_down = effective_trade <= 0
    
        if price_up and oi_up and trade_imb_up:
            state = "EXPANSION"
        elif oi_up and (price_down or price_flat):
            state = "OI_PRICE_DIVERGENCE"
        elif price_up and trade_imb_down:
            state = "FLOW_PRICE_DIVERGENCE"
        elif oi_down and price_down:
            state = "OI_UNWIND"
        elif price_up and (oi_down or oi_flat):
            state = "PRICE_ONLY_EXPANSION"
        else:
            state = "MIXED"
            
    return {
        "state": state,
        "price_basis": price_basis,
        "oi_basis": oi_basis,
        "trade_basis": trade_basis,
        "regime_information_basis": info_basis,
        "effective_price_change": effective_price,
        "effective_oi_change": effective_oi,
        "effective_trade_measure": effective_trade
    }
