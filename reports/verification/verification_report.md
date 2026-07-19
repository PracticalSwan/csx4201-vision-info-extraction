# Rotation Pipeline Verification

Overall: PASS
Profile: full

- [PASS] raw-integrity: raw counts, size, and sampled hashes are unchanged
- [PASS] page-manifest-exists: C:\Assumption University\CSX4201\Project\data\metadata\page_manifest.csv
- [PASS] split-manifest-exists: C:\Assumption University\CSX4201\Project\data\metadata\split_manifest.csv
- [PASS] rotation-manifest-exists: C:\Assumption University\CSX4201\Project\data\metadata\rotation_manifest.csv
- [PASS] selected-page-paths-readable: 603/603 readable
- [PASS] prepared-page-provenance: 603/603 provenance-valid
- [PASS] split-leakage: documents, pages, groups, and privacy are isolated
- [PASS] rotation-ids-unique: duplicate IDs=0
- [PASS] rotation-artifacts-valid: 8332/8332 valid
- [PASS] manifest-matches-files: manifest=8332 physical=8332
- [PASS] all-zones-represented: {1: 2083, 4: 2083, 2: 2083, 3: 2083}
- [PASS] zone-balance: {1: 2083, 4: 2083, 2: 2083, 3: 2083}
- [PASS] boundary-angles: {'validation': 16, 'test': 16}
- [PASS] private-name-leaks: no private filenames in committable code, tests, docs, config, or public artifacts
- [PASS] no-generated-files-under-raw: hits=0
- [PASS] complete-pipeline-artifacts: all required feature/model/evaluation artifacts exist
- [PASS] feature-caches-valid: four splits, dimension=1957, finite and hash-matched
- [PASS] train-only-fit-provenance: scaler, PCA, and K-Means provenance is public train-only
- [PASS] model-artifacts-compatible: typed maintenance artifacts and version-neutral inference parameters reload; k=4, parity, and one-to-one mapping verified
- [PASS] evaluation-outputs-valid: mapped, clustering, boundary, private aggregate, and exact-angle outputs validated
