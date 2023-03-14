# IREE E2E Test Artifacts Models

This directory contains the definitions of source models for e2e tests
(including the benchmark suites). Each source type of models is defined in
`<source_type>_models.py`

## Adding a new model

1.  Upload the source model file to the GCS bucket gs://iree-model-artifacts.
    -   You can ask IREE team members for help if you don't have access.
2.  Register a unique model ID in
    [build_tools/python/e2e_test_framework/unique_ids.py](/build_tools/python/e2e_test_framework/unique_ids.py).
3.  Define a new model with GCS URL and model ID in `<source_type>_models.py`.
4.  Optionally add the model to a model group in
    [build_tools/python/e2e_test_framework/models/model_groups.py](/build_tools/python/e2e_test_framework/models/model_groups.py).
