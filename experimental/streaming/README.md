# Stream HAL - Stream-based Compute Abstraction Layer for IREE

This experimental library provides a unified stream-based compute abstraction layer that supports multiple APIs (CUDA, HIP) on top of IREE HAL. This allows applications using CUDA or HIP APIs to run on any IREE-supported hardware backend including NVIDIA GPUs, AMD GPUs, Vulkan devices, and CPUs.

## Architecture

The library has a three-layer architecture:

```
┌──────────────────────────────────────────────┐
│           Application Layer                  │
│   (CUDA API)              (HIP API)          │
├──────────────────────────────────────────────┤
│         API Compatibility Layer              │
│   cuda_api.c              hip_api.c          │
├──────────────────────────────────────────────┤
│       Stream HAL Internal Layer              │
│  context.c  device.c  memory.c  module.c     │
│  stream.c   event.c   peer.c    init.c       │
├──────────────────────────────────────────────┤
│            IREE HAL Layer                    │
│   (Device abstraction, memory, execution)    │
└──────────────────────────────────────────────┘
```

### Layer Responsibilities

1. **API Compatibility Layer** (`cuda_api.c`, `hip_api.c`)
   - Implements CUDA/HIP Driver API functions
   - Converts API-specific types to internal types
   - Maps error codes between APIs

2. **Stream HAL Internal Layer**
   - Core implementation using `iree_hal_streaming_*` namespace
   - All functions return `iree_status_t` for consistent error handling
   - Platform-agnostic stream-based compute model
   - Files:
     - `context.c` - Context management
     - `device.c` - Device enumeration and properties
     - `memory.c` - Memory allocation and transfers
     - `module.c` - Kernel module loading
     - `stream.c` - Stream/queue management
     - `event.c` - Synchronization events
     - `peer.c` - Peer-to-peer operations
     - `init.c` - Global initialization

3. **IREE HAL Layer**
   - Hardware abstraction provided by IREE
   - Supports multiple backends (CUDA, HIP/ROCm, Vulkan, Metal, CPU)

## Features

- **Multi-Device Support**: Enumerate and use multiple devices across different HAL drivers
- **Cross-Platform Execution**: Run CUDA/HIP code on any IREE backend
- **Unified API**: Single internal implementation supports both CUDA and HIP
- **Raw Binary Loading**: Load pre-compiled kernels (PTX, AMDGCN, SPIR-V, CPU objects)
- **P2P Memory Access**: Cross-device memory transfers when supported
- **Stream-based Execution**: Asynchronous command submission model

## Building

### Bazel Build System

IREE uses Bazel as the primary build system with auto-generated CMake files.

When making changes to the build configuration:
1. Edit the appropriate `BUILD.bazel` file
2. Run the bazel-to-cmake converter to regenerate `CMakeLists.txt`:
   ```bash
   python build_tools/bazel_to_cmake/bazel_to_cmake.py
   ```

#### Core Stream HAL Library
```bash
# Build the core Stream HAL library (internal layer only).
bazel build //experimental/streaming

# Run tests.
bazel test //experimental/streaming/test/...
```

#### API Compatibility Layers
The CUDA and HIP APIs can be built independently:

```bash
# Build CUDA API compatibility layer.
bazel build //experimental/streaming/binding/cuda

# Build HIP API compatibility layer.
bazel build //experimental/streaming/binding/hip

# Build all API layers together.
bazel build //experimental/streaming/binding/...
```

### CMake Build System (Auto-Generated)

The `CMakeLists.txt` files are auto-generated from `BUILD.bazel`. Do not edit them directly.

#### Core Stream HAL Library
```bash
# Configure IREE with Stream HAL support.
cmake -B build/ -S ../.. \
  -DIREE_BUILD_EXPERIMENTAL_HAL_STREAMING=ON

# Build the core Stream HAL library.
cmake --build build/ --target iree_experimental_streaming_streaming

# Compile test kernels for all targets.
cd experimental/streaming/kernels && ./compile_kernels.sh
```

#### API Compatibility Layers
Build the API layers independently:

```bash
# Build CUDA API compatibility layer
cmake --build build/ --target iree_experimental_streaming_binding_cuda_cuda

# Build HIP API compatibility layer
cmake --build build/ --target iree_experimental_streaming_binding_hip_hip

# Build both API layers
cmake --build build/ --target iree_experimental_streaming_binding_cuda_cuda iree_experimental_streaming_binding_hip_hip
```

## Usage Examples

### HIP API Example (test/hip_api_demo.c)

```c
#include "experimental/streaming/binding/hip/api.h"

#define HIP_CHECK(call)                                                 \
  do {                                                                  \
    hipError_t err = (call);                                            \
    if (err != hipSuccess) {                                            \
      fprintf(stderr, "HIP error at %s:%d: %s returned %d\n", __FILE__, \
              __LINE__, #call, err);                                    \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

int main(int argc, char** argv) {
  const char* module_path = argv[1];  // e.g., "kernels/compiled/vector_add.hsaco"
  const int n = 1024;
  const size_t size = n * sizeof(float);

  // 1. Initialize HIP
  HIP_CHECK(hipInit(0));

  // 2. Get device and create context
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  char device_name[256];
  HIP_CHECK(hipDeviceGetName(device_name, sizeof(device_name), device));
  printf("Using device: %s\n", device_name);

  hipCtx_t context;
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  // 3. Load module and get kernel function
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, module_path));

  hipFunction_t vector_add_func;
  HIP_CHECK(hipModuleGetFunction(&vector_add_func, module, "vector_add"));

  // 4. Allocate host memory
  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);

  // Initialize input vectors
  for (int i = 0; i < n; i++) {
    h_A[i] = i * 2.0f;
    h_B[i] = i * 3.0f;
  }

  // 5. Allocate device memory
  hipDeviceptr_t d_A, d_B, d_C;
  HIP_CHECK(hipMalloc(&d_A, size));
  HIP_CHECK(hipMalloc(&d_B, size));
  HIP_CHECK(hipMalloc(&d_C, size));

  // 6. Copy input data to device
  HIP_CHECK(hipMemcpyHtoD(d_A, h_A, size));
  HIP_CHECK(hipMemcpyHtoD(d_B, h_B, size));

  // 7. Launch kernel
  const int threads_per_block = 256;
  const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

  unsigned int n_param = n;
  void* kernel_params[] = {&d_A, &d_B, &d_C, &n_param, NULL};

  HIP_CHECK(hipModuleLaunchKernel(
      vector_add_func,
      blocks_per_grid, 1, 1,  // grid dimensions
      threads_per_block, 1, 1,  // block dimensions
      0,                      // shared memory
      NULL,                   // stream
      kernel_params,          // kernel parameters
      NULL));                 // extra

  // 8. Synchronize and get results
  HIP_CHECK(hipCtxSynchronize());
  HIP_CHECK(hipMemcpyDtoH(h_C, d_C, size));

  // 9. Verify results
  int errors = 0;
  for (int i = 0; i < n; i++) {
    float expected = h_A[i] + h_B[i];  // should be i * 5.0f
    if (fabsf(expected - h_C[i]) > 1e-5f) {
      errors++;
    }
  }
  printf("Result: %s\n", errors == 0 ? "✓ SUCCESS" : "✗ FAILURE");

  // 10. Cleanup
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(context));

  return errors == 0 ? 0 : 1;
}
```

### CUDA API Example

```c
#include "experimental/streaming/binding/cuda/api.h"

// CUDA follows the same pattern - just replace hip* with cu* functions
// hipError_t → CUresult, hipDevice_t → CUdevice, etc.
CUcontext context;
cuInit(0);
cuCtxCreate(&context, 0, device);
// ... rest follows the same structure as HIP example
```

### Multi-Device Example

```c
// Enumerate all devices.
int device_count;
cuDeviceGetCount(&device_count);

// Create contexts for each device.
CUcontext contexts[device_count];
for (int i = 0; i < device_count; i++) {
    CUdevice device;
    cuDeviceGet(&device, i);
    cuCtxCreate(&contexts[i], 0, device);
}

// Check P2P capabilities.
int can_access;
cuDeviceCanAccessPeer(&can_access, 0, 1);
if (can_access) {
    cuCtxSetCurrent(contexts[0]);
    cuCtxEnablePeerAccess(contexts[1], 0);
}

// Perform P2P transfer.
cuMemcpyPeer(d_dst, contexts[1], d_src, contexts[0], size);
```

## Testing

```bash
# Run single-device tests
./build/experimental/streaming/stream_hal_test

# Run multi-device tests
./build/experimental/streaming/stream_hal_multi_test

# Test specific backend
STREAM_HAL_TEST_TARGET=amdgpu ./stream_hal_test
STREAM_HAL_TEST_TARGET=vulkan ./stream_hal_test
STREAM_HAL_TEST_TARGET=cpu ./stream_hal_test
```

## Environment Variables

- `STREAM_HAL_TEST_TARGET`: Force specific backend for testing (amdgpu, vulkan, cpu)
- `STREAM_HAL_DEVICE_ORDINAL`: Default device ordinal to use
- `STREAM_HAL_LOG_LEVEL`: Logging verbosity (0=silent, 5=verbose)

## Internal API

The Stream HAL internal layer provides a clean, status-based API:

```c
// All internal functions use iree_hal_streaming_ prefix and return iree_status_t
iree_status_t iree_hal_streaming_context_create(
    iree_hal_streaming_device_entry_t* device_entry,
    unsigned int flags,
    iree_allocator_t host_allocator,
    iree_hal_streaming_context_t** out_context);

iree_status_t iree_hal_streaming_memory_allocate_device(
    iree_hal_streaming_context_t* context,
    size_t size,
    unsigned int flags,
    iree_hal_streaming_allocation_t** out_allocation);

iree_status_t iree_hal_streaming_launch_kernel(
    iree_hal_streaming_symbol_t* symbol,  // Symbol must be of type FUNCTION
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_stream_t* stream);

// Symbol lookup functions
iree_status_t iree_hal_streaming_module_symbol(
    iree_hal_streaming_module_t* module,
    const char* name,
    iree_hal_streaming_symbol_type_t expected_type,
    iree_hal_streaming_symbol_t** out_symbol);

iree_status_t iree_hal_streaming_module_function(
    iree_hal_streaming_module_t* module,
    const char* name,
    iree_hal_streaming_symbol_t** out_function);  // Returns FUNCTION type symbol

iree_status_t iree_hal_streaming_module_global(
    iree_hal_streaming_module_t* module,
    const char* name,
    iree_hal_streaming_deviceptr_t* out_device_ptr,
    iree_device_size_t* out_size);  // Returns GLOBAL type symbol's device address
```

### (Rough) Type Mapping

| CUDA/HIP Type | Stream HAL Internal Type | IREE HAL Type |
|---------------|-------------------------|---------------|
| `CUcontext`/`hipCtx_t` | `iree_hal_streaming_context_t` | `iree_hal_device_t` |
| `CUmodule`/`hipModule_t` | `iree_hal_streaming_module_t` | `iree_hal_executable_t` |
| `CUfunction`/`hipFunction_t` | `iree_hal_streaming_symbol_t` | Symbol (type=`FUNCTION`) |
| `CUstream`/`hipStream_t` | `iree_hal_streaming_stream_t` | `iree_hal_device_queue_*` |
| `CUdeviceptr`/`hipDeviceptr_t` | `iree_hal_streaming_deviceptr_t` | `iree_hal_buffer_t` |
| `CUevent`/`hipEvent_t` | `iree_hal_streaming_event_t` | `iree_hal_semaphore_t` |
| `CUgraph`/`hipGraph_t` | `iree_hal_streaming_graph_t` | `iree_hal_command_buffer_t` |

**Note**: `iree_hal_streaming_symbol_t` is a unified type that can represent functions, global variables, or data sections.

## API Support Status

The following table shows the implementation status of CUDA and HIP APIs in Stream HAL. Functions are grouped by category and show their current support level.

**Legend:**
- ✅ Fully Supported - Complete implementation
- ⚠️ Partially Supported - Basic functionality works, some features missing
- ❌ Not Supported - Not yet implemented
- 🚫 Won't Support - Will not be implemented (incompatible with HAL model)

### Initialization & Driver APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuInit](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html) | [hipInit](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/initialization_and_version.html#_CPPv47hipInitj) | ✅ | |
| [cuDriverGetVersion](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html) | [hipDriverGetVersion](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/initialization_and_version.html#_CPPv419hipDriverGetVersionPi) | ✅ | |
| cuHALDeinit | hipHALDeinit (HAL extension) | ✅ | |

### Device Management APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuDeviceGet](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html) | hipDeviceGet | ✅ | |
| [cuDeviceGetCount](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html) | [hipGetDeviceCount](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html#_CPPv417hipGetDeviceCountPi) | ✅ | |
| [cuDeviceGetName](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html) | hipDeviceGetName | ✅ | |
| [cuDeviceGetUuid](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html) | hipDeviceGetUuid | ✅ | |
| [cuDeviceTotalMem](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html) | hipDeviceTotalMem | ✅ | |
| [cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html) | [hipDeviceGetAttribute](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html#_CPPv421hipDeviceGetAttributePi20hipDeviceAttribute_ti) | ✅ | |
| | [hipGetDeviceProperties](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html#_CPPv422hipGetDevicePropertiesP15hipDeviceProp_ti) | ✅ | |
| [cuDeviceCanAccessPeer](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html) | [hipDeviceCanAccessPeer](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/peer_to_peer_device_memory_access.html#_CPPv422hipDeviceCanAccessPeerPiii) | ✅ | |
| [cuDeviceGetP2PAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html) | hipDeviceGetP2PAttribute | ✅ | |
| | [hipDeviceSynchronize](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html#_CPPv420hipDeviceSynchronizev) | ✅ | |
| | [hipDeviceReset](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html#_CPPv414hipDeviceResetv) | ✅ | |
| [cuDeviceGetByPCIBusId](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html) | | ❌ | |
| [cuDeviceGetPCIBusId](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html) | | ❌ | |
| [cuDevicePrimaryCtxRetain](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) | [hipDevicePrimaryCtxRetain](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv425hipDevicePrimaryCtxRetainP8hipCtx_t11hipDevice_t) | ✅ | |
| [cuDevicePrimaryCtxRelease](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) | [hipDevicePrimaryCtxRelease](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv426hipDevicePrimaryCtxRelease11hipDevice_t) | ✅ | |
| [cuDevicePrimaryCtxSetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) | [hipDevicePrimaryCtxSetFlags](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv427hipDevicePrimaryCtxSetFlags11hipDevice_tj) | ✅ | |
| [cuDevicePrimaryCtxGetState](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) | [hipDevicePrimaryCtxGetState](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv427hipDevicePrimaryCtxGetState11hipDevice_tPjPi) | ✅ | |
| [cuDevicePrimaryCtxReset](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) | [hipDevicePrimaryCtxReset](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv424hipDevicePrimaryCtxReset11hipDevice_t) | ✅ | |

### Context Management APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuCtxCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipCtxCreate](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv412hipCtxCreateP8hipCtx_tj11hipDevice_t) | ✅ | |
| [cuCtxDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipCtxDestroy](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv413hipCtxDestroy8hipCtx_t) | ✅ | |
| [cuCtxPushCurrent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipCtxPushCurrent](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv417hipCtxPushCurrent8hipCtx_t) | ✅ | |
| [cuCtxPopCurrent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipCtxPopCurrent](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv416hipCtxPopCurrentP8hipCtx_t) | ✅ | |
| [cuCtxSetCurrent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipCtxSetCurrent](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv416hipCtxSetCurrent8hipCtx_t) | ✅ | |
| [cuCtxGetCurrent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipCtxGetCurrent](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv416hipCtxGetCurrentP8hipCtx_t) | ✅ | |
| [cuCtxGetDevice](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipCtxGetDevice](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv415hipCtxGetDeviceP11hipDevice_t) | ✅ | |
| [cuCtxGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | | ✅ | |
| [cuCtxSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipCtxSynchronize](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/context_management.html#_CPPv417hipCtxSynchronizev) | ✅ | |
| [cuCtxSetLimit](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipDeviceSetLimit](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/device_management.html#_CPPv417hipDeviceSetLimit10hipLimit_t6size_t) | ✅ | |
| [cuCtxGetLimit](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | [hipDeviceGetLimit](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/device_management.html#_CPPv417hipDeviceGetLimitP6size_t10hipLimit_t) | ✅ | |
| [cuCtxGetApiVersion](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | | ✅ | |
| [cuCtxGetStreamPriorityRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html) | | ✅ | |
| [cuCtxEnablePeerAccess](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html) | [hipCtxEnablePeerAccess](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/context_management.html#_CPPv422hipCtxEnablePeerAccess8hipCtx_tj) | ✅ | |
| [cuCtxDisablePeerAccess](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html) | [hipCtxDisablePeerAccess](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/context_management.html#_CPPv423hipCtxDisablePeerAccess8hipCtx_t) | ✅ | |

### Module Management APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuModuleLoad](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) | [hipModuleLoad](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html#_CPPv413hipModuleLoadP11hipModule_tPKc) | ✅ | |
| [cuModuleLoadData](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) | [hipModuleLoadData](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html#_CPPv417hipModuleLoadDataP11hipModule_tPKv) | ✅ | |
| [cuModuleLoadDataEx](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) | [hipModuleLoadDataEx](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html#_CPPv419hipModuleLoadDataExP11hipModule_tPKvjP10hipJitOption_t) | ✅ | |
| [cuModuleLoadFatBinary](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) | | ✅ | |
| [cuModuleUnload](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) | [hipModuleUnload](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html#_CPPv415hipModuleUnload11hipModule_t) | ✅ | |
| [cuModuleGetFunction](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) | [hipModuleGetFunction](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html#_CPPv419hipModuleGetFunctionP13hipFunction_t11hipModule_tPKc) | ✅ | |
| [cuModuleGetGlobal](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) | [hipModuleGetGlobal](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html#_CPPv417hipModuleGetGlobalP14hipDeviceptr_tP6size_t11hipModule_tPKc) | ✅ | |

### Memory Management APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuMemGetInfo](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemGetInfo](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv413hipMemGetInfoP6size_tP6size_t) | ✅ | |
| [cuMemAlloc](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMalloc](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv49hipMallocPP14hipDeviceptr_t6size_t) | ✅ | |
| [cuMemAllocPitch](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMallocPitch](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv414hipMallocPitchPPvP6size_t6size_t6size_t) | ✅ | |
| [cuMemFree](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipFree](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv47hipFree14hipDeviceptr_t) | ✅ | |
| [cuMemGetAddressRange](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemGetAddressRange](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv420hipMemGetAddressRangeP14hipDeviceptr_tP6size_t14hipDeviceptr_t) | ✅ | |
| [cuMemAllocHost](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMallocHost](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv413hipMallocHostPPv6size_t) | ✅ | |
| [cuMemFreeHost](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipFreeHost](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv411hipFreeHostPv) | ✅ | |
| [cuMemHostAlloc](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipHostAlloc](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv412hipHostAllocPPv6size_tj) | ✅ | |
| [cuMemHostGetDevicePointer](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipHostGetDevicePointer](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv423hipHostGetDevicePointerP14hipDeviceptr_tPvj) | ✅ | |
| [cuMemHostGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipHostGetFlags](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv415hipHostGetFlagsPjPv) | ✅ | |
| [cuMemAllocManaged](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMallocManaged](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/managed_memory.html#_CPPv416hipMallocManagedPP14hipDeviceptr_t6size_tj) | ❌ | |
| [cuMemHostRegister](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipHostRegister](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv415hipHostRegisterPv6size_tj) | ✅ | |
| [cuMemHostUnregister](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipHostUnregister](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv417hipHostUnregisterPv) | ✅ | |
| | [hipMemPtrGetInfo](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv416hipMemPtrGetInfoPvP6size_t) | ✅ | |

### Memory Transfer APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuMemcpy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpy](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv49hipMemcpyPvPKv6size_t21hipMemcpyKind_t) | ✅ | |
| [cuMemcpyPeer](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpyPeer](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/peer_to_peer_device_memory_access.html#_CPPv413hipMemcpyPeer14hipDeviceptr_t8hipCtx_t14hipDeviceptr_t8hipCtx_t6size_t) | ✅ | |
| [cuMemcpyHtoD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpyHtoD](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv413hipMemcpyHtoD14hipDeviceptr_tPv6size_t) | ✅ | |
| [cuMemcpyDtoH](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpyDtoH](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv413hipMemcpyDtoHPv14hipDeviceptr_t6size_t) | ✅ | |
| [cuMemcpyDtoD](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpyDtoD](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv413hipMemcpyDtoD14hipDeviceptr_t14hipDeviceptr_t6size_t) | ✅ | |
| [cuMemcpyAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpyAsync](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv414hipMemcpyAsyncPvPKv6size_t21hipMemcpyKind_t11hipStream_t) | ✅ | |
| [cuMemcpyPeerAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | | ✅ | |
| [cuMemcpyHtoDAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpyHtoDAsync](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv418hipMemcpyHtoDAsync14hipDeviceptr_tPv6size_t11hipStream_t) | ✅ | |
| [cuMemcpyDtoHAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpyDtoHAsync](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv418hipMemcpyDtoHAsyncPv14hipDeviceptr_t6size_t11hipStream_t) | ✅ | |
| [cuMemcpyDtoDAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemcpyDtoDAsync](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv418hipMemcpyDtoDAsync14hipDeviceptr_t14hipDeviceptr_t6size_t11hipStream_t) | ✅ | |
| | [hipMemcpyWithStream](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv419hipMemcpyWithStreamPvPKv6size_t21hipMemcpyKind_t11hipStream_t) | ✅ | |

### Memory Set APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| | [hipMemset](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv49hipMemsetPvii6size_t) | ✅ | |
| | [hipMemsetAsync](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv414hipMemsetAsyncPvii6size_t11hipStream_t) | ✅ | |
| [cuMemsetD8](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemsetD8](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv411hipMemsetD814hipDeviceptr_th6size_t) | ✅ | |
| [cuMemsetD16](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemsetD16](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv412hipMemsetD1614hipDeviceptr_tt6size_t) | ✅ | |
| [cuMemsetD32](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemsetD32](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv412hipMemsetD3214hipDeviceptr_ti6size_t) | ✅ | |
| [cuMemsetD8Async](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemsetD8Async](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv416hipMemsetD8Async14hipDeviceptr_th6size_t11hipStream_t) | ✅ | |
| [cuMemsetD16Async](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemsetD16Async](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv417hipMemsetD16Async14hipDeviceptr_tt6size_t11hipStream_t) | ✅ | |
| [cuMemsetD32Async](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMemsetD32Async](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv417hipMemsetD32Async14hipDeviceptr_ti6size_t11hipStream_t) | ✅ | |

### IPC Memory APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuIpcGetMemHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | | ❌ | |
| [cuIpcOpenMemHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | | ❌ | |
| [cuIpcCloseMemHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | | ❌ | |

### Kernel Execution APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuLaunchKernel](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | [hipModuleLaunchKernel](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/execution_control.html#_CPPv421hipModuleLaunchKernel13hipFunction_tjjjjjjj11hipStream_tPPvPPv) | ✅ | |
| [cuLaunchCooperativeKernel](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | [hipModuleLaunchCooperativeKernel](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/cooperative_groups_reference.html#_CPPv432hipModuleLaunchCooperativeKernel13hipFunction_tjjjjjjj11hipStream_tPPv) | ⚠️ | No multi-block sync |
| [cuLaunchHostFunc](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | [hipLaunchHostFunc](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/launch_api.html#_CPPv417hipLaunchHostFunc11hipStream_t11hipHostFn_tPv) | ✅ | |
| [cuFuncGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | [hipFuncGetAttribute](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/execution_control.html#_CPPv419hipFuncGetAttributePi21hipFunction_attribute13hipFunction_t) | ✅ | |
| | [hipFuncGetAttributes](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/execution_control.html#_CPPv420hipFuncGetAttributesP17hipFuncAttributesPKv) | ✅ | |
| [cuFuncSetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | [hipFuncSetAttribute](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/execution_control.html#_CPPv419hipFuncSetAttributePKv16hipFuncAttributei) | ✅ | |
| [cuFuncSetCacheConfig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | [hipFuncSetCacheConfig](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/execution_control.html#_CPPv421hipFuncSetCacheConfigPKv14hipFuncCache_t) | ✅ | |
| [cuFuncSetSharedMemConfig](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | [hipFuncSetSharedMemConfig](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/execution_control.html#_CPPv425hipFuncSetSharedMemConfigPKv18hipSharedMemConfig) | ✅ | |

### Occupancy APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuOccupancyMaxActiveBlocksPerMultiprocessor](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html) | [hipModuleOccupancyMaxActiveBlocksPerMultiprocessor](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/occupancy.html#_CPPv450hipModuleOccupancyMaxActiveBlocksPerMultiprocessorPi13hipFunction_ti6size_t) | ✅ | |
| [cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html) | [hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/occupancy.html#_CPPv459hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsPi13hipFunction_ti6size_tj) | ✅ | |
| [cuOccupancyMaxPotentialBlockSize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html) | [hipModuleOccupancyMaxPotentialBlockSize](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/occupancy.html#_CPPv439hipModuleOccupancyMaxPotentialBlockSizePiPi13hipFunction_t6size_ti) | ✅ | |
| | [hipModuleOccupancyMaxPotentialBlockSizeWithFlags](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/occupancy.html#_CPPv448hipModuleOccupancyMaxPotentialBlockSizeWithFlagsPiPi13hipFunction_t6size_tij) | ✅ | |

### Stream Management APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuStreamCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamCreate](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv415hipStreamCreateP11hipStream_t) | ✅ | |
| | [hipStreamCreateWithFlags](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv424hipStreamCreateWithFlagsP11hipStream_tj) | ✅ | |
| [cuStreamCreateWithPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamCreateWithPriority](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv427hipStreamCreateWithPriorityP11hipStream_tji) | ✅ | |
| [cuStreamGetPriority](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamGetPriority](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv420hipStreamGetPriority11hipStream_tPi) | ✅ | |
| [cuStreamGetFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamGetFlags](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv417hipStreamGetFlags11hipStream_tPj) | ✅ | |
| [cuStreamGetCtx](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamGetDevice](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv418hipStreamGetDevice11hipStream_tP11hipDevice_t) | ✅ | |
| [cuStreamWaitEvent](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamWaitEvent](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv418hipStreamWaitEvent11hipStream_t10hipEvent_tj) | ✅ | |
| [cuStreamQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamQuery](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv414hipStreamQuery11hipStream_t) | ✅ | |
| [cuStreamSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamSynchronize](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv420hipStreamSynchronize11hipStream_t) | ✅ | |
| [cuStreamDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamDestroy](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv416hipStreamDestroy11hipStream_t) | ✅ | |
| [cuStreamCopyAttributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | | ✅ | |

### Event Management APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuEventCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | [hipEventCreate](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html#_CPPv414hipEventCreateP10hipEvent_t) | ✅ | |
| | [hipEventCreateWithFlags](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html#_CPPv423hipEventCreateWithFlagsP10hipEvent_tj) | ✅ | |
| [cuEventRecord](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | [hipEventRecord](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html#_CPPv414hipEventRecord10hipEvent_t11hipStream_t) | ✅ | |
| [cuEventQuery](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | [hipEventQuery](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html#_CPPv413hipEventQuery10hipEvent_t) | ✅ | |
| [cuEventSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | [hipEventSynchronize](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html#_CPPv419hipEventSynchronize10hipEvent_t) | ✅ | |
| [cuEventDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | [hipEventDestroy](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html#_CPPv415hipEventDestroy10hipEvent_t) | ✅ | |
| [cuEventElapsedTime](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | [hipEventElapsedTime](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html#_CPPv419hipEventElapsedTimePf10hipEvent_t10hipEvent_t) | ✅ | |
| [cuIpcGetEventHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | | ❌ | |
| [cuIpcOpenEventHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | | ❌ | |

### Unified Memory APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuMemAdvise](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) | [hipMemAdvise](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/managed_memory.html#_CPPv412hipMemAdvisePKv6size_t19hipMemoryAdvise_ti) | ❌ | |
| [cuMemPrefetchAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) | [hipMemPrefetchAsync](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/managed_memory.html#_CPPv419hipMemPrefetchAsyncPKv6size_ti11hipStream_t) | ❌ | |
| [cuPointerGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) | [hipPointerGetAttribute](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv422hipPointerGetAttributePv21hipPointer_attribute14hipDeviceptr_t) | ⚠️ | Limited attributes |
| [cuPointerSetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) | [hipPointerSetAttribute](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv422hipPointerSetAttributePKv21hipPointer_attribute14hipDeviceptr_t) | ⚠️ | Limited attributes |
| [cuPointerGetAttributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) | [hipPointerGetAttributes](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html#_CPPv423hipPointerGetAttributesjP21hipPointer_attributePPvP14hipDeviceptr_t) | ⚠️ | Limited attributes |
| [cuMemRangeGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) | [hipMemRangeGetAttribute](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/managed_memory.html#_CPPv423hipMemRangeGetAttributePv6size_t18hipMemRangeAttributePKv6size_t) | ⚠️ | Limited attributes |
| [cuMemRangeGetAttributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html) | [hipMemRangeGetAttributes](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/managed_memory.html#_CPPv424hipMemRangeGetAttributesPPvP6size_tP18hipMemRangeAttributePKv6size_t6size_t) | ⚠️ | Limited attributes |

### Graph APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuGraphCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphCreate](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv414hipGraphCreateP10hipGraph_tj) | ✅ | |
| [cuGraphDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphDestroy](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv415hipGraphDestroy10hipGraph_t) | ✅ | |
| [cuGraphInstantiate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphInstantiate](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv419hipGraphInstantiateP14hipGraphExec_t10hipGraph_tP14hipGraphNode_tPc6size_t) | ✅ | |
| [cuGraphInstantiateWithFlags](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphInstantiateWithFlags](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv428hipGraphInstantiateWithFlagsP14hipGraphExec_t10hipGraph_ty) | ✅ | |
| [cuGraphExecDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphExecDestroy](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv419hipGraphExecDestroy14hipGraphExec_t) | ✅ | |
| [cuGraphLaunch](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphLaunch](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv414hipGraphLaunch14hipGraphExec_t11hipStream_t) | ✅ | |
| [cuGraphExecUpdate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphExecUpdate](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv418hipGraphExecUpdate14hipGraphExec_t10hipGraph_tP14hipGraphNode_tP24hipGraphExecUpdateResult) | ❌ | |
| [cuGraphAddKernelNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphAddKernelNode](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv421hipGraphAddKernelNodeP14hipGraphNode_t10hipGraph_tPK14hipGraphNode_t6size_tPK19hipKernelNodeParams) | ✅ | |
| [cuGraphAddMemcpyNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphAddMemcpyNode](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv421hipGraphAddMemcpyNodeP14hipGraphNode_t10hipGraph_tPK14hipGraphNode_t6size_tPK16hipMemcpy3DParms) | ✅ | |
| [cuGraphAddMemsetNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphAddMemsetNode](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv421hipGraphAddMemsetNodeP14hipGraphNode_t10hipGraph_tPK14hipGraphNode_t6size_tPK15hipMemsetParams) | ✅ | |
| [cuGraphAddHostNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphAddHostNode](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv419hipGraphAddHostNodeP14hipGraphNode_t10hipGraph_tPK14hipGraphNode_t6size_tPK17hipHostNodeParams) | ✅ | |
| [cuGraphAddEmptyNode](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | [hipGraphAddEmptyNode](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv420hipGraphAddEmptyNodeP14hipGraphNode_t10hipGraph_tPK14hipGraphNode_t6size_t) | ✅ | |
| [cuStreamBeginCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamBeginCapture](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv420hipStreamBeginCapture11hipStream_t20hipStreamCaptureMode) | ✅ | |
| [cuStreamEndCapture](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamEndCapture](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv418hipStreamEndCapture11hipStream_tP10hipGraph_t) | ✅ | |
| [cuStreamIsCapturing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamIsCapturing](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv419hipStreamIsCapturing11hipStream_tP23hipStreamCaptureStatus) | ✅ | |
| [cuStreamGetCaptureInfo](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamGetCaptureInfo](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv422hipStreamGetCaptureInfo11hipStream_tP23hipStreamCaptureStatusPy) | ✅ | |
| [cuStreamUpdateCaptureDependencies](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html) | [hipStreamUpdateCaptureDependencies](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html#_CPPv433hipStreamUpdateCaptureDependencies11hipStream_tP14hipGraphNode_t6size_tj) | ✅ | |

### Memory Pool APIs

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| [cuMemPoolCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolCreate](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv416hipMemPoolCreateP12hipMemPool_tPK15hipMemPoolProps) | ⚠️ | |
| [cuMemPoolDestroy](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolDestroy](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv417hipMemPoolDestroy12hipMemPool_t) | ✅ | |
| [cuMemPoolSetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolSetAttribute](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv422hipMemPoolSetAttribute12hipMemPool_t14hipMemPoolAttrPv) | ⚠️ | |
| [cuMemPoolGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolGetAttribute](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv422hipMemPoolGetAttribute12hipMemPool_t14hipMemPoolAttrPv) | ⚠️ | |
| [cuMemPoolSetAccess](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolSetAccess](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv419hipMemPoolSetAccess12hipMemPool_tPK16hipMemAccessDesc6size_t) | ❌ | |
| [cuMemPoolGetAccess](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolGetAccess](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv419hipMemPoolGetAccessP17hipMemAccessFlags12hipMemPool_tP14hipMemLocation) | ❌ | |
| [cuMemPoolTrimTo](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolTrimTo](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv416hipMemPoolTrimTo12hipMemPool_t6size_t) | ❌ | |
| [cuMemPoolExportToShareableHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolExportToShareableHandle](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv433hipMemPoolExportToShareableHandlePv12hipMemPool_t26hipMemAllocationHandleTypej) | ❌ | |
| [cuMemPoolImportFromShareableHandle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolImportFromShareableHandle](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv435hipMemPoolImportFromShareableHandleP12hipMemPool_tPv26hipMemAllocationHandleTypej) | ❌ | |
| [cuMemPoolExportPointer](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolExportPointer](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv423hipMemPoolExportPointerP23hipMemPoolPtrExportDataPv) | ❌ | |
| [cuMemPoolImportPointer](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMemPoolImportPointer](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv423hipMemPoolImportPointerPPv12hipMemPool_tP23hipMemPoolPtrExportData) | ❌ | |
| [cuDeviceSetMemPool](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | hipDeviceSetMemPool | ✅ | |
| [cuDeviceGetMemPool](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | hipDeviceGetMemPool | ✅ | |
| [cuDeviceGetDefaultMemPool](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | hipDeviceGetDefaultMemPool | ✅ | |
| [cuMemAllocAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipMallocAsync](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv414hipMallocAsyncPPv6size_t11hipStream_t) | ❌ | |
| [cuMemAllocFromPoolAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM__POOL.html) | [hipMallocFromPoolAsync](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv422hipMallocFromPoolAsyncPPv6size_t12hipMemPool_t11hipStream_t) | ❌ | |
| [cuMemFreeAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html) | [hipFreeAsync](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/memory_management/stream_ordered_memory_allocator.html#_CPPv412hipFreeAsyncPv11hipStream_t) | ❌ | |

### Error Handling APIs (HIP only)

| CUDA | HIP | Status | Issues |
|------|-----|--------|--------|
| | [hipGetErrorString](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/error_handling.html#_CPPv417hipGetErrorString10hipError_t) | ✅ | |
| | [hipGetErrorName](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/error_handling.html#_CPPv415hipGetErrorName10hipError_t) | ✅ | |
| | [hipGetLastError](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/error_handling.html#_CPPv415hipGetLastErrorv) | ✅ | |
| | [hipPeekAtLastError](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/error_handling.html#_CPPv418hipPeekAtLastErrorv) | ✅ | |

## Contributing

See [PLAN.md](PLAN.md) for implementation details and roadmap.

## License

Apache 2.0 with LLVM Exceptions. See LICENSE file in IREE root.
