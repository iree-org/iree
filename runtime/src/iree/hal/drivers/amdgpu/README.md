# AMD GPU HAL Driver (`amdgpu`)

**NOTE**: the code is the authoritative documentation source. This document is an overview of the implementation and should be treated as informational only. See the linked files for details.

## Quick Start

Configure CMake with the following options:

```sh
-DIREE_BUILD_COMPILER=ON
-DIREE_TARGET_BACKEND_ROCM=ON
-DIREE_HAL_DRIVER_AMDGPU=ON
-DIREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGETS=gfx1100
-DIREE_HIP_TEST_TARGET_CHIP=gfx1100
```

Substitute the architecture with your own. See [therock_amdgpu_targets.cmake](https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake#L44) for a list of common targets. Future changes will include family support matching that file.

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

The `IREE_HAL_AMDGPU_DEVICE_LIBRARY_TARGETS` CMake variable can be set to a list of target architectures to build the library for and bundle into the AMDGPU HAL library. Architectures not built into the library will fail to instantiate the driver at runtime.
