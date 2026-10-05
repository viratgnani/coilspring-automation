# CoilBrain Research Engine V1

The Research Engine aligns completed CoilSpring signals with their corresponding microstructure snapshots and forward price observations at standard horizons (30s, 1m, 2m, 5m, 10m, 30m, 1h). This analytical layer evaluates whether microstructure trajectories correlate with subsequent price expansion, independent of the signal score.

## Architecture & Schema

The Research Engine is an analytical module that reads from `signals`, `price_observations`, and `microstructure_snapshots`, and writes output into two tables:

### 1. `research_signal_features`
Contains signal-level features and the **T0 baseline microstructure**.
- **MFE / MAE**: Calculated over the signal's full active window (typically 4 hours).
- **T0 Baseline**: The Microstructure snapshot captured exactly at or immediately after the signal timestamp (alignment rules below).

### 2. `research_horizon_outcomes`
Contains outcome rows for every signal across defined horizons (`30s, 60s, 120s, 300s, 600s, 1800s, 3600s`).
- **Price Outcome**: Forward price return at that exact horizon.
- **Microstructure Outcome**: Snapshotted Microstructure state at that horizon.
- **Microstructure Deltas**: Percentage or absolute changes in microstructure (OI, spread, imbalances, etc.) from the T0 baseline.
- **Alignment Metadata**: How closely the observation and snapshots aligned with the target horizon time.

## Alignment & Handling Rules

### T0 Alignment
To find the initial T0 Microstructure state:
1. The engine scans for snapshots within `[-5s, +10s]` relative to the signal `t0_timestamp`.
2. It chooses the closest snapshot.
3. In a tie, it prefers the pre-signal snapshot.
4. If no snapshot exists within this window, the T0 microstructure baseline is completely `NULL`.

### Horizon Alignment
For each forward horizon:
1. The target time is calculated as `t0_timestamp + horizon_s`.
2. The allowed tolerance is `± max(60s, horizon_s * 10%)`. This prevents capturing a data point too far in the future due to downtime.
3. The closest price observation and closest microstructure snapshot within the tolerance window are captured.

### Missing Data
- **Missing values are preserved as `NULL`**. Zero-filling is strictly avoided.
- A missing price observation means the `return_pct` will be `NULL`.
- A missing snapshot (at T0 or the horizon) means microstructure deltas will be `NULL`.
- True zeros (like `0` trade count) are preserved.

## Usage (CLI)

The Research Engine can be triggered manually via the CLI:

```bash
# Generate analytical outcomes for all COMPLETE signals not yet in the research tables
python -m coilbrain.research --generate

# View a full trajectory timeline for a specific signal
python -m coilbrain.research --signal-id <signal_id>

# View timeline for all signals belonging to a symbol (e.g. MYXUSDT)
python -m coilbrain.research --symbol MYXUSDT

# Print bucketed descriptive stats and runner comparisons
# (Default runner threshold is MFE >= 2.0%)
python -m coilbrain.research --summary --runner-threshold 2.5
```
