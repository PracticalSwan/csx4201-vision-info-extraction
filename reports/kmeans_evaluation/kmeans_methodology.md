# K-Means Rotation Methodology

K-Means is fitted without labels on public training features only.
The scaler and optional PCA are also fitted only on that same training scope.
Cluster IDs are arbitrary, so a 4 by 4 training count matrix is mapped
one-to-one to Zones 1-4 using Hungarian assignment after K-Means fitting.

- Clusters: 4
- Training rows: 5600
- PCA enabled: True
- Output dimension: 128
- Inertia: 6771246.500000
- Iterations: 16
- Mapping: {'0': 1, '1': 2, '2': 4, '3': 3}

Centroid-distance confidence is a heuristic margin, not a calibrated probability.
