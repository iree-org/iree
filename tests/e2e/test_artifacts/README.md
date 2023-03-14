IREE E2E Test Artifacts Suite
===============================

IREE E2E test artifacts suite is a collection of artifacts for e2e tests and
benchmarking, which usually depend on external models and module compilations.

Update the test artifacts
-------------------------

All the test artifacts are defined by the python modules. To add/remove/update
these artifacts, modify the related python modules and regenerate the CMake
files with the command below:

```sh
build_tools/scripts/generate_cmake_files.sh
```

Here are the places to find the definitions of the artifacts:
- Model sources: `build_tools/python/e2e_test_framework/models`
- Benchmarks: `build_tools/python/benchmark_suites`
