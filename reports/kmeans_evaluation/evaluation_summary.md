# K-Means Rotation Evaluation

Positive rotation angles are counterclockwise and zones use half-open 90-degree intervals.
Classification metrics use the fixed training-derived cluster mapping. Clustering
metrics use raw cluster IDs. No artifact is refitted during evaluation.

- K-Means inertia: 6771246.500000
- Cluster sizes: {'0': 1112, '1': 1114, '2': 1686, '3': 1688}

## Metrics by split

- private_test: accuracy=0.5000, macro F1=0.3654, ARI=0.4793, NMI=0.6174
- test: accuracy=0.3792, macro F1=0.3772, ARI=0.0871, NMI=0.1035
- train: accuracy=0.5000, macro F1=0.4947, ARI=0.3422, NMI=0.5038
- validation: accuracy=0.3781, macro F1=0.3702, ARI=0.0886, NMI=0.1021

Private-test results are aggregate-only. No private filename, preview,
source path, or per-document prediction is written.
