"""Application configuration loaded from environment variables / .env file."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is loaded into real os.environ
load_dotenv()


class Settings(BaseSettings):
    """All runtime settings for CoilBrain.

    Values are read (in priority order) from:
      1. Real environment variables
      2. A `.env` file in the working directory
      3. The defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── Server ────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    # ── Service identity ──────────────────────────────────────────────────────
    service_name: str = "coilbrain"
    log_level: str = "INFO"

    # ── Persistence ───────────────────────────────────────────────────────────
    # SQLite file path relative to the working directory (or absolute).
    # Swap for postgresql+asyncpg://... later without code changes.
    database_url: str = "coilbrain.db"

    # ── Signal Observatory ────────────────────────────────────────────────────
    # Total observation window per signal (hours).
    obs_window_hours: float = 4.0

    # Phase 1: T+0 → T+obs_phase1_duration_s  — dense polling.
    obs_phase1_duration_s: float = 3600.0    # 1 hour
    obs_phase1_interval_s: float = 30.0      # every 30 seconds

    # Phase 2: T+phase1_duration → end of window — sparse polling.
    obs_phase2_interval_s: float = 300.0     # every 5 minutes

    # How often the picker loop wakes to find new OPEN signals (seconds).
    obs_picker_interval_s: float = 30.0

    # How often the symbol registry is refreshed from Binance (hours).
    obs_registry_refresh_hours: float = 6.0

    # Binance USDM Futures public endpoints (override for testing/mocking).
    obs_binance_exchange_info_url: str = (
        "https://fapi.binance.com/fapi/v1/exchangeInfo"
    )
    obs_binance_price_url: str = (
        "https://fapi.binance.com/fapi/v1/ticker/price"
    )

    # HTTP timeout for observatory price-fetch calls (seconds).
    obs_http_timeout_s: float = 8.0

    # ── Microstructure Recorder ───────────────────────────────────────────────
    # How often each symbol's microstructure is snapshotted (seconds).
    ms_poll_interval_s: float = 30.0

    # Approximate recording window for documentation/monitoring purposes.
    # NOTE: the recorder does NOT independently enforce this window.  The
    # symbol loop exits when the Observatory marks a signal COMPLETE (which
    # happens after obs_window_hours = 4h).  This setting is informational
    # only and must equal obs_window_hours to avoid confusion.
    ms_window_hours: float = 4.0

    # How often the recorder's symbol-picker wakes (seconds).
    ms_picker_interval_s: float = 30.0

    # Maximum distinct symbols to record simultaneously (rate-limit guard).
    ms_max_symbols: int = 20

    # Warn when Binance used-weight-per-minute exceeds this fraction of 2400.
    ms_rate_limit_warn_pct: float = 0.80

    # Number of consecutive missed polls before a stale-data warning is logged.
    ms_stale_threshold_multiplier: float = 3.0

    # Order book depth limit (levels per side).  limit<=100 costs weight 2.
    ms_book_depth_limit: int = 20

    # Toggle individual data sources.
    # Trades are disabled by default: aggTrades costs weight 20 per call.
    ms_enable_oi: bool = True
    ms_enable_funding: bool = True
    ms_enable_book: bool = True
    ms_enable_trades: bool = True

    # Binance USDM Futures microstructure endpoints.
    ms_binance_oi_url: str = "https://fapi.binance.com/fapi/v1/openInterest"
    ms_binance_premium_url: str = "https://fapi.binance.com/fapi/v1/premiumIndex"
    ms_binance_depth_url: str = "https://fapi.binance.com/fapi/v1/depth"
    ms_binance_aggtrades_url: str = "https://fapi.binance.com/fapi/v1/aggTrades"
    ms_http_timeout_s: float = 8.0

    # How many days of microstructure snapshots to retain.
    # The picker loop runs a purge once per day; set 0 to disable purging.
    ms_retain_days: int = 30


# Module-level singleton — import this everywhere you need settings.
settings = Settings()

