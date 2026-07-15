# Exact-Angle Estimation Evaluation

Each angle search is restricted to the mapped K-Means zone. Positive
angles are counterclockwise; the applied correction is the signed
negative estimate. Circular errors use the shortest distance on [0,360).

Low-confidence estimates remain in primary error metrics and are marked
explicitly. Hard failures have no angle and are reported via failure rate.

## Public results

- test: n=944, estimated=928, circular MAE=89.8192, median=90.0000, within 5 degrees=0.3330
- validation: n=976, estimated=976, circular MAE=89.6714, median=90.0000, within 5 degrees=0.3494

## Private test

- Aggregate-only sample count: 812
- Circular MAE: 90.0037
- Failure rate: 0.0099
- No private filename, path, identifier, preview, or row prediction was written.

Correction validation scores are pixel-derived orientation scores before
and after applying `correction_angle = -estimated_angle`; they are not a
second fitted model or a calibrated confidence probability.
