# Organizer checklist

1. Download the Kaggle Road Sign Detection dataset and unzip it into `data/raw/RoadSign/` (folders `images/` and `annotations/`).
2. `python scripts/prepare_data.py --raw data/raw/RoadSign` (writes private staging images)
3. `python scripts/build_graphs.py` (needs `scikit-image`; writes the public graph files and `data/private/test_labels.csv`)
4. `python scripts/generate_keys.py` (first time only), add the GitHub secrets `SUBMISSION_PRIVATE_KEY` and `LABELS_KEY`.
5. `python scripts/encrypt_labels.py`, commit `encryption/public_key.pem` and `encryption/test_labels.enc`.
6. Run `python scripts/baseline.py`, record the scores in section 8, and time it on a CPU.
7. `python scripts/make_release.py` and upload the zip to a GitHub Release `v1.0`.
8. Enable GitHub Pages (branch `main`, folder `/docs`) and Actions write permission.
9. Do a dry run with a test PR from a second account.
