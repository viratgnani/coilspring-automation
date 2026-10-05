"""CoilBrain notification client for Coiled Spring Scanner v2.

Isolation contract
------------------
* This module is the ONLY file that knows about CoilBrain.
* It must NEVER raise an exception visible to the caller.
* It must NEVER block the scanner loop.
* A durable SQLite outbox guarantees every signal survives CoilBrain outages.
* A background delivery worker retries indefinitely with bounded exponential
  back-off; signals are NEVER permanently discarded due to connectivity failure.

Usage
-----
    import coilbrain_client
    coilbrain_client.notify(result)   # fire-and-forget; always persists locally

Environment
-----------
    COILBRAIN_URL       Full URL of CoilBrain POST /signals endpoint.
                        Example: http://YOUR_COILBRAIN_HOST:8000/signals
                        Empty string (default) -> integration silently disabled
                        (outbox is still written so signals survive being re-enabled).

    COILBRAIN_OUTBOX    Path to the SQLite outbox file.
                        Default: coilbrain_outbox.db (relative to cwd).

Delivery semantics
------------------
* A UUID4 is generated ONCE when a signal is first enqueued.
  It is stored in the outbox and sent as X-Idempotency-Key on every POST.
  It is NEVER regenerated during retry.
* CoilBrain uses X-Idempotency-Key as the canonical signal_id, making
  every retry safe: duplicate POSTs always return the existing signal.
* Signals are retried INDEFINITELY until delivered; they never become
  permanently DEAD due to connectivity or server errors.
* DEAD state is reserved exclusively for malformed/corrupt payloads
  that cannot be delivered regardless of server state.
* Back-off schedule (seconds): 5, 10, 20, 40, 80, 160, 320, 640, 1280,
  then 1280 repeatedly. The attempt counter is used only for logging.

Outbox schema (outbox_items)
-----------------------------
    id              TEXT PRIMARY KEY  -- client-generated UUID4 (idempotency key)
    payload_json    TEXT              -- full JSON-encoded payload
    queued_at       REAL              -- unix timestamp of first enqueue
    attempts        INTEGER           -- delivery attempts so far (for logging only)
    last_attempt_at REAL              -- unix timestamp of last attempt (or NULL)
    state           TEXT              -- PENDING | DELIVERED | DEAD
    last_error      TEXT              -- last error message (or NULL)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
import urllib.request
from typing import Any, Dict, Optional

log = logging.getLogger("coiled_spring.coilbrain_client")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT_S: float = 5.0
_WORKER_POLL_S: float = 5.0          # how often the worker wakes to check outbox
_LOG_ALERT_ATTEMPTS: int = 10        # log a warning when this many attempts exceeded
_BASE_BACKOFF_S: float = 5.0        # initial back-off
_MAX_BACKOFF_S: float = 1280.0      # cap back-off at ~21 min, then repeat forever

# States
_PENDING = "PENDING"
_DELIVERED = "DELIVERED"
_DEAD = "DEAD"          # ONLY for permanently corrupt/undeliverable payloads

# ---------------------------------------------------------------------------
# Module-level worker state (one worker thread per process lifetime)
# ---------------------------------------------------------------------------

_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_shutdown_event = threading.Event()

# ---------------------------------------------------------------------------
# Outbox DB helpers
# ---------------------------------------------------------------------------

def _outbox_path() -> str:
    return os.environ.get("COILBRAIN_OUTBOX", "coilbrain_outbox.db").strip() or "coilbrain_outbox.db"


def _get_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")   # BLOCKER 5: crash-safe durability
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outbox_items (
            id              TEXT PRIMARY KEY,
            payload_json    TEXT NOT NULL,
            queued_at       REAL NOT NULL,
            attempts        INTEGER NOT NULL DEFAULT 0,
            last_attempt_at REAL,
            state           TEXT NOT NULL DEFAULT 'PENDING',
            last_error      TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_outbox_state
        ON outbox_items (state, last_attempt_at)
    """)
    conn.commit()


def _enqueue(conn: sqlite3.Connection, idempotency_key: str, payload: Dict[str, Any]) -> None:
    """Insert signal into outbox. If the key already exists, do nothing (idempotent)."""
    conn.execute(
        """
        INSERT OR IGNORE INTO outbox_items
            (id, payload_json, queued_at, attempts, state)
        VALUES (?, ?, ?, 0, 'PENDING')
        """,
        (idempotency_key, json.dumps(payload), time.time()),
    )
    conn.commit()


def _backoff_for(attempts: int) -> float:
    """Return seconds to wait before the next attempt given prior attempt count.

    Caps at _MAX_BACKOFF_S and repeats indefinitely — signals are never abandoned.
    """
    wait = _BASE_BACKOFF_S * (2 ** min(attempts, 30))   # prevent overflow for huge counts
    return min(wait, _MAX_BACKOFF_S)


def _due_items(conn: sqlite3.Connection) -> list:
    """Return PENDING items that are due for a delivery attempt."""
    now = time.time()
    rows = conn.execute(
        """
        SELECT id, payload_json, attempts, last_attempt_at
        FROM outbox_items
        WHERE state = 'PENDING'
        ORDER BY queued_at ASC
        """,
    ).fetchall()

    due = []
    for row in rows:
        id_, payload_json, attempts, last_attempt_at = row
        if last_attempt_at is None:
            due.append(row)
        else:
            backoff = _backoff_for(attempts - 1)
            if (now - last_attempt_at) >= backoff:
                due.append(row)
    return due


def _mark_delivered(conn: sqlite3.Connection, idempotency_key: str) -> None:
    conn.execute(
        "UPDATE outbox_items SET state='DELIVERED', last_attempt_at=? WHERE id=?",
        (time.time(), idempotency_key),
    )
    conn.commit()


def _mark_failed(
    conn: sqlite3.Connection,
    idempotency_key: str,
    error: str,
    attempts: int,
    permanent: bool = False,
) -> None:
    """Mark a failed delivery attempt.

    permanent=True  -> DEAD (corrupt payload, will never succeed)
    permanent=False -> stays PENDING for retry (transient connectivity/server error)
    """
    new_state = _DEAD if permanent else _PENDING
    conn.execute(
        """
        UPDATE outbox_items
        SET attempts=?, last_attempt_at=?, last_error=?, state=?
        WHERE id=?
        """,
        (attempts, time.time(), error[:500], new_state, idempotency_key),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Delivery worker
# ---------------------------------------------------------------------------

def _deliver_one(url: str, idempotency_key: str, payload_json: str) -> None:
    """Attempt one HTTP POST. Raises on failure.

    Sends X-Idempotency-Key so CoilBrain can detect and safely handle retries.
    """
    body = payload_json.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Idempotency-Key": idempotency_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        status = resp.status
    if status not in (200, 201, 202):
        raise RuntimeError(f"Unexpected HTTP status {status}")


def _worker_loop(outbox_path: str) -> None:
    """Background thread: drain the outbox, retrying indefinitely with back-off."""
    log.info("CoilBrain outbox worker started (outbox=%s)", outbox_path)
    try:
        conn = _get_conn(outbox_path)
        _ensure_schema(conn)
    except Exception as exc:
        log.error("CoilBrain outbox worker failed to open DB: %s", exc)
        return

    while not _shutdown_event.is_set():
        url = os.environ.get("COILBRAIN_URL", "").strip()
        if not url:
            _shutdown_event.wait(timeout=_WORKER_POLL_S)
            continue

        try:
            due = _due_items(conn)
        except Exception as exc:
            log.warning("Outbox read error: %s", exc)
            _shutdown_event.wait(timeout=_WORKER_POLL_S)
            continue

        for row in due:
            if _shutdown_event.is_set():
                break
            idempotency_key, payload_json, attempts, _ = row

            # Validate payload is parseable before attempting delivery.
            # Corrupt payloads are permanently undeliverable -> DEAD.
            try:
                json.loads(payload_json)
            except (json.JSONDecodeError, TypeError) as exc:
                log.error(
                    "[DEAD] key=%s corrupted payload, marking DEAD permanently: %s",
                    idempotency_key, exc,
                )
                _mark_failed(conn, idempotency_key, f"CORRUPTED: {exc}", attempts, permanent=True)
                continue

            new_attempts = attempts + 1
            if new_attempts > 1:
                log.info(
                    "[RETRYING] key=%s attempt=%d url=%s",
                    idempotency_key, new_attempts, url,
                )
            if new_attempts > _LOG_ALERT_ATTEMPTS:
                log.warning(
                    "[ALERT] key=%s has exceeded %d attempts — CoilBrain may be offline",
                    idempotency_key, _LOG_ALERT_ATTEMPTS,
                )

            try:
                _deliver_one(url, idempotency_key, payload_json)
                _mark_delivered(conn, idempotency_key)
                log.info("[RECORDED] key=%s delivered to CoilBrain", idempotency_key)
            except Exception as exc:
                err_str = str(exc)
                # NEVER permanent for connectivity/HTTP errors — retry indefinitely
                _mark_failed(conn, idempotency_key, err_str, new_attempts, permanent=False)
                backoff = _backoff_for(new_attempts - 1)
                log.warning(
                    "[FAILED] key=%s attempt=%d, next retry in %.0fs: %s",
                    idempotency_key, new_attempts, backoff, err_str,
                )

        _shutdown_event.wait(timeout=_WORKER_POLL_S)

    log.info("CoilBrain outbox worker stopped.")


def _ensure_worker_running(outbox_path: str) -> None:
    """Start the background delivery worker thread if not already running."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _shutdown_event.clear()
        t = threading.Thread(
            target=_worker_loop,
            args=(outbox_path,),
            daemon=True,
            name="coilbrain-outbox-worker",
        )
        t.start()
        _worker_thread = t


# ---------------------------------------------------------------------------
# Payload construction (unchanged schema)
# ---------------------------------------------------------------------------

def build_payload(r) -> Dict[str, Any]:
    """Map an AnalysisResult to the CoilBrain POST /signals body.

    Field mapping (exact -- no fabrication):
        symbol              r.symbol
        exchange            "binance"              (V2 scans Binance USDM)
        trigger             "COILED_SPRING_V2"     (constant)
        grade               r.grade
        score               r.score
        signal              "PUMP"                 (constant per spec)
        oi_status           "OI_PRESENT" if any flag contains "OI_PRESENT"
                            else "UNKNOWN"
        price               r.last_close
        change_24h_pct      r.change_24h
        volume_24h_usd      r.volume_24h_usd
        breakout_pct        r.trigger_close_pct_above_high
        body_volume_spike   r.trigger_body_multiple
        volume_spike        r.trigger_vol_multiple
        flat_period_days    r.flat_candles / 24
        signal_timestamp    time.time()  (stamped at notify() call time)
    """
    oi_status = "OI_PRESENT" if any("OI_PRESENT" in f for f in r.flags) else "UNKNOWN"

    return {
        "symbol":            r.symbol,
        "exchange":          "binance",
        "trigger":           "COILED_SPRING_V2",
        "grade":             r.grade,
        "score":             float(r.score),
        "signal":            "PUMP",
        "oi_status":         oi_status,
        "price":             float(r.last_close),
        "change_24h_pct":    float(r.change_24h),
        "volume_24h_usd":    float(r.volume_24h_usd),
        "breakout_pct":      float(r.trigger_close_pct_above_high),
        "body_volume_spike": float(r.trigger_body_multiple),
        "volume_spike":      float(r.trigger_vol_multiple),
        "flat_period_days":  float(r.flat_candles) / 24.0,
        "signal_timestamp":  time.time(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def notify(r) -> None:
    """Persist signal to local outbox and ensure the delivery worker is running.

    A UUID4 idempotency key is generated ONCE here and stored with the outbox
    row. It is never regenerated during retries. notify() returns immediately;
    the scanner loop is never blocked.
    """
    try:
        # Stamp timestamp and generate idempotency key NOW
        payload = build_payload(r)
        idempotency_key = str(uuid.uuid4())   # BLOCKER 1: UUID4, generated once
        outbox_path = _outbox_path()

        # Persist to outbox before any HTTP attempt
        conn = _get_conn(outbox_path)
        _ensure_schema(conn)
        _enqueue(conn, idempotency_key, payload)
        conn.close()

        log.info(
            "[QUEUED] key=%s symbol=%s price=%.6f",
            idempotency_key, r.symbol, payload["price"],
        )

        _ensure_worker_running(outbox_path)

    except Exception as exc:
        log.warning("coilbrain_client.notify failed unexpectedly (scanner unaffected): %s", exc)


def _post(url: str, payload: Dict[str, Any], symbol: str) -> None:
    """Legacy single-shot POST helper -- retained for backwards-compatibility
    with existing tests. Production delivery goes through the outbox worker.
    """
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            status = resp.status
        log.info("CoilBrain notified | symbol=%s status=%d", symbol, status)
    except Exception as exc:
        log.warning(
            "CoilBrain notify failed (symbol=%s): %s -- scanner unaffected.",
            symbol,
            exc,
        )


# ---------------------------------------------------------------------------
# Diagnostic command
# ---------------------------------------------------------------------------

def outbox_status() -> Dict[str, Any]:
    """Return a dict with outbox health metrics (safe to call anytime)."""
    try:
        outbox_path = _outbox_path()
        conn = _get_conn(outbox_path)
        _ensure_schema(conn)

        pending_count = conn.execute(
            "SELECT COUNT(*) FROM outbox_items WHERE state='PENDING'"
        ).fetchone()[0]

        oldest_pending = conn.execute(
            "SELECT queued_at FROM outbox_items WHERE state='PENDING' ORDER BY queued_at ASC LIMIT 1"
        ).fetchone()

        last_delivered = conn.execute(
            "SELECT last_attempt_at FROM outbox_items WHERE state='DELIVERED' ORDER BY last_attempt_at DESC LIMIT 1"
        ).fetchone()

        dead_count = conn.execute(
            "SELECT COUNT(*) FROM outbox_items WHERE state='DEAD'"
        ).fetchone()[0]

        total_delivered = conn.execute(
            "SELECT COUNT(*) FROM outbox_items WHERE state='DELIVERED'"
        ).fetchone()[0]

        conn.close()

        now = time.time()
        oldest_age_s = None
        if oldest_pending:
            oldest_age_s = now - oldest_pending[0]

        last_delivery_ago_s = None
        if last_delivered and last_delivered[0]:
            last_delivery_ago_s = now - last_delivered[0]

        return {
            "pending_count": pending_count,
            "dead_count": dead_count,
            "total_delivered": total_delivered,
            "oldest_pending_age_s": oldest_age_s,
            "last_successful_delivery_ago_s": last_delivery_ago_s,
            "outbox_path": outbox_path,
        }
    except Exception as exc:
        return {"error": str(exc)}


def print_status() -> None:
    """Print a human-readable outbox status report."""
    s = outbox_status()
    if "error" in s:
        print(f"Outbox status ERROR: {s['error']}")
        return

    pending = s["pending_count"]
    dead = s["dead_count"]
    delivered = s["total_delivered"]
    oldest = s["oldest_pending_age_s"]
    last_ok = s["last_successful_delivery_ago_s"]
    path = s["outbox_path"]

    oldest_str = f"{oldest:.0f}s ago" if oldest is not None else "N/A"
    last_ok_str = f"{last_ok:.0f}s ago" if last_ok is not None else "never"

    print(f"CoilBrain Outbox [{path}]")
    print(f"  Pending signals      : {pending}")
    print(f"  Dead (corrupt only)  : {dead}")
    print(f"  Total delivered      : {delivered}")
    print(f"  Oldest pending age   : {oldest_str}")
    print(f"  Last successful del. : {last_ok_str}")


if __name__ == "__main__":
    print_status()
