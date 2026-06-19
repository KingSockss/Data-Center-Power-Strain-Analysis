# Model V0.1

Simple Ridge regression for one-hour-ahead DOM real-time LMP.

Target: `dom_rt_lmp` for target hour `s`.

Features:

- LMP lag: `s-1`
- LMP daily lag: `s-24`
- Load lag and daily lag
- Congestion lag and daily lag
- Temperature lag and daily lag
- Temperature forecast if a forecast temperature column exists
- Hour/day/month cyclic encodings and weekend flag
- 6-hour and 24-hour trailing simple moving averages for LMP, load, and congestion

Split:

- Train: first nine months
- Test: final three months

Implementation notes:

- Ridge is fit with a closed-form NumPy implementation and standardized features.
- Error is `prediction - actual`, so positive bias means overprediction.
- WAPE is reported as a percent.
- MASE uses the first nine months and a 24-hour seasonal naive denominator.
- Directional accuracy compares the predicted next-hour move against the actual next-hour move relative to the prior observed LMP.

Run:

```bash
python Model_V0.1/run_ridge_model.py
```

Outputs:

- `Model_V0.1/outputs/model_dataset.parquet`
- `Model_V0.1/outputs/model_dataset.csv`
- `Model_V0.1/outputs/ridge_predictions.csv`
- `Model_V0.1/outputs/ridge_coefficients.csv`
- `Model_V0.1/outputs/model_comparison_metrics.csv`
- `Model_V0.1/outputs/true_vs_ridge_forecast.html`
- `Model_V0.1/outputs/true_vs_all_models.html`
