# Package validation

- Python source files: syntax-compiled successfully.
- All five source CSV files are present.
- The quick pipeline was executed successfully in the build environment, exercising both tracks, model fitting, temporal validation, conformal calibration and model serialization.
- Smoke-run artifacts were removed from this distributable so they cannot be mistaken for final project results or pre-trained production models.
- Re-run `RUN_QUICK_TEST.bat` after extraction to reproduce a smoke test. Run `RUN_FULL_BENCHMARK.bat` only after reviewing the protocol.
- This package does not contain final validated results. Do not cite sandbox smoke-run numbers in a report.
