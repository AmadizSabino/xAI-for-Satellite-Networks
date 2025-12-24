# SLA Breach — Final Summary
**Run date (UTC):** 2025-11-25T20:25:01+00:00

**Model ranking quality (Validation)**
- Average Precision (AP): 0.460
- Selected t for precision≈0.80: 0.996

**Early Warning**
- Lead target: 2 minute(s) before breach

**Eventization preset (final)**
- EMA span: 4
- ON/OFF: 0.945512068271637 / 0.765512068271637
- Debounce K/M: 7/11
- Min duration / Refractory: 90s / 240s

**Test metrics (this window)**
- FAR: 0.00 alerts/hour
- Median detection delay: 1640.0 s
- Median lead time: NA s
- Event Precision / Recall: 1.000 / 0.030
- Pred events / GT events: 1 / 33

**Notes**
- t chosen on validation (no test leakage).
- Eventization tuned via small preset sweep and locked; results are reproducible from config.json.
