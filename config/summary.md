# Beam Handover Anomalies — Final Summary
**Run date (UTC):** 2025-11-25T21:03:21+00:00

**Beam inference**
- MODEM *_OUT* normalized vector -> PCA(6) -> KMeans(K=2)
- Handover = beam_id change (smoothed ≥60s)

**Anomaly rules (trained on TRAIN)**
- Post-handover throughput drop > 0.042
- Ping-pong within 300s
- Recovery time > 45s

**Model (early warning)**
- XGB on rolling throughput + beam dynamics; target=anomalous handover in 2 min

**Validation**
- AP: 0.105 | τ (val): 0.946

**Eventization preset (final)**
- EMA span: 4
- ON/OFF: 0.796 / 0.676
- Debounce K/M: 3/5
- Min duration / Refractory: 120s / 60s

**Test metrics (this window)**
- FAR: 0.010 alerts/hour
- Median detection delay: 84655.0 s
- Median lead time: 215285.0 s
- Event Precision / Recall: 0.500 / 0.012
- Pred events / GT events: 2 / 84

**Notes**
- All thresholds learned on TRAIN only; timelines split 60/20/20.
