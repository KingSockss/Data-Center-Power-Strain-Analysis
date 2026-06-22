# Model V0.1.1

Simple Ridge regression for one-day-ahead DOM real-time LMP.

Target: `dom_rt_lmp(t+24)` from forecast origin hour `t`.

This version intentionally changes only the forecast horizon from Model V0.1. Features remain lagged/trailing values anchored at the forecast origin, so the model does not use values from the future target window.

Split:

- Train: first nine months by target timestamp
- Test: final three months by target timestamp

Baselines:

- `persistence_origin_lmp`: `dom_rt_lmp(t)`
- `daily_persistence_origin_minus_24`: `dom_rt_lmp(t-24)`
- `pjm_day_ahead_lmp_for_target`: `dom_da_lmp(t+24)`

Run:

```bash
python Model_V0.1.1/run_ridge_model_t_plus_24.py
```

Outputs:

- `Model_V0.1.1/outputs/model_dataset.parquet`
- `Model_V0.1.1/outputs/model_dataset.csv`
- `Model_V0.1.1/outputs/ridge_t_plus_24_predictions.csv`
- `Model_V0.1.1/outputs/ridge_coefficients.csv`
- `Model_V0.1.1/outputs/model_comparison_metrics.csv`
- `Model_V0.1.1/outputs/true_vs_ridge_t_plus_24_forecast.html`
- `Model_V0.1.1/outputs/true_vs_all_t_plus_24_models.html`
