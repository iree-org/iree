# AMD GPU HAL Driver (`amdgpu`)

**NOTE**: the code is the authoritative documentation source. This document is an overview of the implementation and should be treated as informational only. See the linked files for details.

## Quick Start

### CMake

Configure CMake with the following options:

```sh
-DIREE_BUILD_COMPILER=ON
-DIREE_TARGET_BACKEND_ROCM=ON
-DIREE_HAL_DRIVER_AMDGPU=ON
-DIREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGETS=all
-DIREE_ROCM_TEST_TARGET_CHIP=gfx1100
```

### Bazel

Build tools with the AMDGPU runtime driver registered and with device artifacts
compiled for your local GPU architecture:

```sh
iree-bazel-build //tools:iree-compile //tools:iree-run-module \
  --iree_drivers=amdgpu,cuda,hip,local-sync,local-task,vulkan \
  --//build_tools/bazel:rocm_test_target=gfx1100
```

The ROCM chip target defaults to `gfx1100`. Override for your hardware:

```sh
iree-bazel-test --//build_tools/bazel:rocm_test_target=gfx942 //runtime/src/iree/hal/drivers/amdgpu/cts/...
```

Substitute the architecture with your own. See [therock_amdgpu_targets.cmake](https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake) for the target and generic family vocabulary mirrored by the embedded device library build.

Use `amdgpu` to specify devices at runtime:

```sh
# Single logical device with all available physical devices:
iree-run-module --device=amdgpu
# Device ordinal 0 (danger, this may change across reboots):
iree-run-module --device=amdgpu:0
# Device with a stable UUID for a device:
iree-run-module --device=amdgpu://GPU-0e12865a3bf5b7ab
# Single logical device with the two devices given by their UUIDs:
iree-run-module --device=amdgpu://GPU-0e12865a3bf5b7ab,GPU-89e8bdf59a10cf6d
# Single logical device with physical devices with ordinals 2 and 3:
ROCR_VISIBLE_DEVICES=2,3 iree-run-module --device=amdgpu
# Two logical devices with two physical devices each:
iree-run-module --device=amdgpu://0,1 --device=amdgpu://2,3
```

Use `amdgpu` to specify the AMDGPU target when compiling programs:

```sh
iree-compile --iree-hal-target-device=amdgpu ...
```

For a direct Bazel-built smoke test, compile with the AMDGPU target device and
run the resulting VMFB with the AMDGPU HAL driver:

```sh
bazel-bin/tools/iree-compile \
  --iree-input-type=stablehlo \
  --iree-hal-target-device=amdgpu \
  --iree-rocm-target=gfx1100 \
  --iree-rocm-bc-dir=bazel-bin/external/_main~iree_extension~amdgpu_device_libs/bitcode \
  tests/e2e/stablehlo_models/mnist_fake_weights.mlir \
  -o=/tmp/mnist_fake_amdgpu.vmfb

bazel-bin/tools/iree-run-module \
  --device=amdgpu \
  --module=/tmp/mnist_fake_amdgpu.vmfb \
  --function=predict \
  --input=1x28x28x1xf32 \
  --expected_output='1x10xf32=0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1'
```

Prefer this explicit two-step flow when debugging AMDGPU target-device behavior.
`iree-run-mlir --device=amdgpu` currently relies on generic device-to-compiler
flag inference and may select the legacy ROCm/HIP target path instead of the
AMDGPU HAL target.

To capture a Tracy trace of the same runtime path, use the Bazel trace wrapper.
The wrapper requires a `tracy-capture` binary in `PATH`, or one supplied through
`IREE_TRACY_CAPTURE`:

```sh
IREE_TRACY_CAPTURE=/path/to/tracy-capture \
  build_tools/bin/iree-bazel-run \
  --trace \
  --trace_name=fake_mnist \
  //tools:iree-run-module \
  --iree_drivers=amdgpu,cuda,hip,local-sync,local-task,vulkan \
  --//build_tools/bazel:rocm_test_target=gfx1100 \
  -- \
  --device=amdgpu \
  --module=/tmp/mnist_fake_amdgpu.vmfb \
  --function=predict \
  --input=1x28x28x1xf32 \
  --expected_output='1x10xf32=0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1'
```

## Build Notes

### HSA/ROCR Dependency

We maintain a fork of the HSA headers required for compilation as [third_party/hsa-runtime-headers/](https://github.com/iree-org/hsa-runtime-headers). This fork may also contain tweaks not yet upstreamed required to use the headers in our build.

We require that at runtime a dynamic library with the name `libhsa-runtime64.so` exists on the path. This can be overridden programmatically when constructing the driver, via the `--amdgpu_libhsa_search_path=` flag if using the command line tools, via the `IREE_HAL_AMDGPU_LIBHSA_PATH` environment variable, or by just adding a directory containing the file to `PATH`.

It's recommended that developers check out a copy of the [ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime) and build it locally in whatever configuration they are using (debug/release/ASAN/etc). This allows for easier debugging and profiling as symbols are present and may be required to get recent features not available in platform installs. Eventually IREE will ship its own copy of the library (directly or indirectly) as part of the install packages such that only a relatively recent AMDGPU driver is required.

See [HSA/ROCR Library](#hsarocr-library) for more information on our usage.

### Device Library Compilation

**Required CMake Options**: `-DIREE_BUILD_COMPILER=ON -DIREE_TARGET_BACKEND_ROCM=ON`

**Top-level Build Target**: `iree_hal_drivers_amdgpu_device_binaries`

Currently IREE's CMake configuration must have the compiler enabled in order to build the runtime including the AMDGPU HAL implementation. This will be made better in the future (allowing for just building what we need instead of the full MLIR stack, using an existing ROCM install, etc). See [Device Library](#device-library) for more information.

The device library should be compiled automatically when building the AMDGPU HAL driver and gets embedded inside the runtime binary so that no additional files are required at runtime.

The `IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGETS` CMake variable defaults to `all`, which embeds LLVM generic ISA code objects covering every currently known AMDGPU device library target. Packagers can set it to a smaller list of exact target architectures, LLVM generic ISA targets, TheRock-style generic target families, or TheRock-style product bundles. Exact targets use the HSA ISA spelling, such as `gfx1100`. LLVM generic ISA targets use spellings such as `gfx11-generic`. Generic families use the TheRock family spelling, such as `gfx110X-all`, and product bundles use spellings such as `dgpu-all` or `igpu-all`. These selectors expand to the smallest known compatible code object set instead of one code object per exact GPU. Architectures not built into the library will fail to instantiate the driver at runtime.

The Bazel build exposes the same selector vocabulary through `//runtime/src/iree/hal/drivers/amdgpu/device/binaries:targets`:

```sh
iree-bazel-build --//runtime/src/iree/hal/drivers/amdgpu/device/binaries:targets=igpu-all //runtime/src/iree/hal/drivers/amdgpu:amdgpu
```

See [`device/binaries/README.md`](device/binaries/README.md) for the target map
update flow, the generated Bazel/CMake/runtime fragments, and the TheRock/LLVM
sources that should be checked when adding support for a new architecture.
