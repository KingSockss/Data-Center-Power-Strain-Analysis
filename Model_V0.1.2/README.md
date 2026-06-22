# Model V0.1.2

Simple Ridge regression for one-day-ahead DOM real-time LMP with two target-hour forecast inputs added.

Target: `dom_rt_lmp(t+24)` from forecast origin hour `t`.

Changes from Model V0.1.1:

- Adds `target_pjm_forecast_load_mw`: `pjm_forecast_load_mw(t+24)`
- Adds `target_dom_da_lmp`: `dom_da_lmp(t+24)`, the PJM day-ahead LMP for the same target hour

All other feature families are inherited from Model V0.1.1.

Important interpretation note: `target_dom_da_lmp` is also one of the comparison baselines. Including it as a Ridge input tests whether the linear model can improve on PJM day-ahead LMP by combining it with lagged LMP, load, congestion, weather, and calendar features.

Run:

```bash
python Model_V0.1.2/run_ridge_model_t_plus_24_with_forecasts.py
```

Outputs:

- `Model_V0.1.2/outputs/model_dataset.parquet`
- `Model_V0.1.2/outputs/model_dataset.csv`
- `Model_V0.1.2/outputs/ridge_t_plus_24_predictions.csv`
- `Model_V0.1.2/outputs/ridge_coefficients.csv`
- `Model_V0.1.2/outputs/model_comparison_metrics.csv`
- `Model_V0.1.2/outputs/true_vs_ridge_t_plus_24_forecast.html`
- `Model_V0.1.2/outputs/true_vs_all_t_plus_24_models.html`
