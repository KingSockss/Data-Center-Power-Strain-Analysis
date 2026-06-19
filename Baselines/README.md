# Baselines

Baseline forecasts for one-hour-ahead DOM real-time LMP.

Target interpretation: for target hour `s`, predict `dom_rt_lmp(s)`.

- `persistence`: `dom_rt_lmp(s-1)`
- `daily_persistence`: `dom_rt_lmp(s-24)`
- `pjm_day_ahead_lmp`: `dom_da_lmp(s)`, the PJM day-ahead market LMP for the same target hour

Metrics are computed on the final three months of the dataset after using the first nine months as the train/reference period.

Metric notes:

- Error is `prediction - actual`, so positive bias means overprediction.
- WAPE is reported as a percent.
- MASE uses the first nine months and a 24-hour seasonal naive denominator.
- Directional accuracy compares the predicted next-hour move against the actual next-hour move relative to the prior observed LMP.

Run:

```bash
python Baselines/run_baselines.py
```

Outputs:

- `Baselines/outputs/baseline_predictions.csv`
- `Baselines/outputs/baseline_metrics.csv`
