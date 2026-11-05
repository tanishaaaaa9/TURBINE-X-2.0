# TURBINE-X Dataset Audit Report

**Audit basis:** the five CSV attachments uploaded in this conversation, read directly from their mounted paths.

**Total rows:** 36,733  
**Columns:** 11  
**Missing values:** 0  
**Non-finite values:** 0  
**Duplicate occurrences within annual files:** 7  
**Cross-year exact duplicate groups:** 0

| File | Bytes | Rows | Columns | Missing | Duplicate rows | SHA256 |
|---|---:|---:|---:|---:|---:|---|
| gt_2011.csv | 562921 | 7411 | 11 | 0 | 0 | `d87ceef9aa59533c…` |
| gt_2012.csv | 577617 | 7628 | 11 | 0 | 0 | `be54b9d0e1a7de40…` |
| gt_2013.csv | 540334 | 7152 | 11 | 0 | 0 | `13c437bb440ec204…` |
| gt_2014.csv | 541009 | 7158 | 11 | 0 | 7 | `c2a03c92c9c3207a…` |
| gt_2015.csv | 556928 | 7384 | 11 | 0 | 0 | `9b08f35fde0d4b13…` |

## Schema

AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO, NOX

All columns parse as float64 in all five files. No categorical, timestamp or identifier columns are present.

## Data-quality interpretation

- Seven duplicate occurrences appear in `gt_2014.csv`, belonging to two repeated-value groups. They are retained in the primary data and excluded only in a sensitivity branch.
- All 11 columns vary; no constant columns were detected.
- The largest absolute correlations are TEY–CDP (0.989), GTEP–CDP (0.978), GTEP–TEY (0.964), TIT–TEY (0.910), TIT–CDP (0.908), and TIT–GTEP (0.874).
- The files contain no row timestamps. Preserve within-file row order and annual file identity; do not infer exact temporal intervals or create lag features.
- This is a single-plant observational dataset. Predictive association does not establish causal control effects or safe operating limits.

## Observed ranges

| Column | Minimum | Maximum |
|---|---:|---:|
| AT | -6.2348 | 37.103 |
| AP | 985.85 | 1036.6 |
| AH | 24.085 | 100.2 |
| AFDP | 2.0874 | 7.6106 |
| GTEP | 17.698 | 40.716 |
| TIT | 1000.8 | 1100.9 |
| TAT | 511.04 | 550.61 |
| TEY | 100.02 | 179.5 |
| CDP | 9.8518 | 15.159 |
| CO | 0.00038751 | 44.103 |
| NOX | 25.905 | 119.91 |