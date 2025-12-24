# Signal Loss – Final Summary
**Run date (UTC):** 2025-11-15T23:49:02Z

**Model ranking quality**
- Validation AP: 0.832  
- Selected τ (target precision=0.90): 0.761

**Eventization (Winner):** Preset L
- ON/OFF: 0.84/0.66
- K/M: 7/11
- Min duration / Refractory: 90s / 240s

**Test results (this period)**
- FAR: 0.97 alerts/hour
- Median detection delay: 1840.0 s
- Event Precision / Recall (from sweep record): 0.899 / 0.220

**Notes**
- Threshold chosen on validation (no test leakage); eventization tuned via sweep, then locked.
- Next: backtest across multiple days and (optionally) calibrate probabilities for τ stability.
