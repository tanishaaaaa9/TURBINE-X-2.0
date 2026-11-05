# Implementation status and research safeguards

This package contains executable conventional-ML pipelines, a Streamlit inference demo, dataset files, audit documents, experiment templates, tracker and project report/blueprint. It does not contain pre-trained/validated models: those are created on the user's computer after running the benchmark.

## Implemented in source
- Strict year-wise CSV loader and schema checks.
- Chronological development (2011-2013), calibration (2014), locked test (2015) split.
- Two research tracks: A (CO/NOx; TEY is an input) and B (CO/NOx/TEY; TEY excluded from inputs).
- Dummy, Ridge, ElasticNet, Random Forest, Extra Trees and HistGradientBoosting conventional ML.
- Independent/multi-output support where appropriate.
- Per-target MAE, RMSE, R2, MedAE; runtime.
- 90% split-conformal marginal intervals calibrated on 2014 and evaluated on 2015.
- Model serialization and a basic Streamlit prediction interface.
- Development-only chronological validation folds and output tables.

## Still requires execution / review
- Run the full benchmark locally and inspect all warnings/results.
- Expand HPO/AutoML only within 2011-2013 development data.
- Add block-bootstrap uncertainty and deeper SHAP/ALE reports after selecting models.
- Validate conformal exchangeability assumptions: temporal distribution shift can invalidate nominal coverage. Report observed 2015 coverage, never promise 90% coverage.
- Supervisor/researcher must verify citations, novelty, interpretation and final paper claims.

## Important
No timestamp exists in the CSV rows. Do not fabricate exact time-of-day, lags, rolling windows or hourly sequences. This is observational data from one plant. No causal claims, operational setpoint recommendations or safety limits are supported.
