# IREE E2E Test Artifacts Suite

IREE E2E test artifacts suite is a collection of artifacts for e2e tests and
benchmarking, which usually depend on external models and module compilations.

## Updating Test Artifacts

All the test artifacts are defined by the python modules. To add/remove/update
these artifacts, modify the related python modules and regenerate the CMake
files with the command below:

```sh
build_tools/scripts/generate_cmake_files.sh
```

Here are the places to find the definitions of the artifacts:

-   Model sources:
    [build_tools/python/e2e_test_framework/models](/build_tools/python/e2e_test_framework/models)
-   Benchmarks:
    [build_tools/python/benchmark_suites](/build_tools/python/benchmark_suites)

## Debugging Build Failures

When an IREE module is failed to build in e2e test artifacts, the error message
will be like:

```
[1/1] Generating /.../e2e_test_artifacts/iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb from MobileNetV3Small_fp32(tflite) [x86_64-cascadelake-linux_gnu-llvm_cpu][default-flags]
FAILED: e2e_test_artifacts/iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb /.../e2e_test_artifacts/iree_MobileNetV3Small_fp32_module_baec9d4086496a94853f349354f87acb8397bf36169134d3269d5803888dcf49/module.vmfb
cd /.../tests/e2e/test_artifacts && /.../tools/iree-compile --output-format=vm-bytecode ...
ninja: build stopped: subcommand failed.
```

The first line shows `Generating ... from <friendly name with the module
architecture and tags>`. You can find the compile flags in
[tests/e2e/test_artifacts/generated_e2e_test_iree_artifacts.cmake](/tests/e2e/test_artifacts/generated_e2e_test_iree_artifacts.cmake)
by searching with it.

> Note that some texts might be truncated from a long output line when shown on
> the console. You can pipe the CMake output to a log file to get full texts.

The model name, architecture, and tags should also lead you to the source that
generates the build rule under
[build_tools/python/benchmark_suites](/build_tools/python/benchmark_suites).
