# Model V0.1.3

Closed-form Ridge regression for one-day-ahead DOM real-time LMP with nonlinear and interaction transforms.

This model inherits all Model V0.1.2 features and adds:

- Target forecast load × lag-1 congestion
- Target forecast load × hour sine
- Target forecast load × hour cosine
- Lag-1 temperature squared
- Lag-1 congestion squared
- Target forecast load squared
- Natural log of target day-ahead LMP
- `(target forecast load × lag-1 congestion)^2 / target day-ahead LMP`

The two requested load×congestion expressions are algebraically identical, so they are represented by one feature rather than duplicated.

For this dataset, all target day-ahead LMP values are positive, so the natural log and denominator are valid. Future rows with nonpositive day-ahead LMP are excluded from this version rather than silently changing the requested formulas.

Run:

```bash
python Model_V0.1.3/run_ridge_model_with_transforms.py
```

The comparison table includes a Model V0.1.2 refit on exactly the same rows, which isolates the effect of the added transforms.

Outputs include:

- `Model_V0.1.3/outputs/model_comparison_metrics.csv`
- `Model_V0.1.3/outputs/ridge_coefficients.csv`
- `Model_V0.1.3/outputs/transform_coefficients.csv`
- `Model_V0.1.3/outputs/transform_definitions.csv`
- `Model_V0.1.3/outputs/true_vs_ridge_t_plus_24_forecast.html`
- `Model_V0.1.3/outputs/true_vs_all_t_plus_24_models.html`
