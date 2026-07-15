# Rotation Feature Extraction

- Strategy: hog_hough
- Version: orientation-v1
- Fixed feature dimension: 1957
- Configuration hash: 888fb4999c985ba0
- NaN values: 0
- Infinite values: 0
- Failed vectors: 0

The default vector concatenates spatial OpenCV HOG, a length-weighted
Hough orientation histogram and line statistics, horizontal and vertical
projection profiles, directional edge densities, and page geometry.
Feature extraction performs no fitting and does not use validation, test,
or private labels to alter the representation.

## Counts by split

- private_test: 812
- test: 944
- train: 5600
- validation: 976
