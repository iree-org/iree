### :link: Artifact Links

```sh
# Compiler and host tools archive:
export INSTALL_DIR_URL="${INSTALL_DIR_GCS_ARTIFACT}"
# Directory of e2e test artifacts:
export E2E_TEST_ARTIFACTS_DIR_URL="${E2E_TEST_ARTIFACTS_GCS_ARTIFACT_DIR}"
# Directory of benchmark tool binaries:
export BENCHMARK_TOOLS_DIR_URL="${BENCHMARK_TOOLS_GCS_ARTIFACT_DIR}"
# Directory of execution benchmark results and traces:
export EXECUTION_BENCHMARK_RESULTS_DIR_URL="${EXECUTION_BENCHMARK_RESULTS_GCS_ARTIFACT_DIR}"
# Compilation benchmark results:
export COMPILATION_BENCHMARK_RESULTS_URL="${COMPILATION_BENCHMARK_RESULTS_GCS_ARTIFACT}"
```

You can list `gs://` directories and download files with [gcloud cli](
https://cloud.google.com/sdk/gcloud/reference/storage).

<details>
<summary>Click to show common usages</summary>

```sh
# Get compile and run flags of benchmarks
gcloud storage cp -r "${E2E_TEST_ARTIFACTS_GCS_ARTIFACT_DIR}/benchmark-flag-dump.txt" /tmp/
```

```sh
# Download MLIR input files and command lines
mkdir /tmp/iree_e2e_test_inputs
gcloud storage cp "${E2E_TEST_ARTIFACTS_GCS_ARTIFACT_DIR}/*" /tmp/iree_e2e_test_inputs
```

```sh
# Download all artifacts (MLIR input files and compiled VMFBs)
mkdir /tmp/iree_e2e_test_artifacts
gcloud storage cp -r "${E2E_TEST_ARTIFACTS_GCS_ARTIFACT_DIR}/*" /tmp/iree_e2e_test_artifacts
```

```sh
# Download execution benchmark results and traces
mkdir /tmp/iree_benchmark_results
gcloud storage cp -r "${EXECUTION_BENCHMARK_RESULTS_GCS_ARTIFACT_DIR}/*" /tmp/iree_benchmark_results
```
</details>

To run benchmarks locally with the CI-built e2e test artifacts, see [IREE Benchmark Suites](
https://iree.dev/developers/performance/benchmark-suites/#3-fetch-the-benchmark-artifacts).
