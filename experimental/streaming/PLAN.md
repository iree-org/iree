# Stream HAL Refactoring Plan

## Overview
Refactor the current CUDA HAL implementation to support both CUDA and HIP APIs through a unified Stream HAL layer. This creates a single implementation that can be accessed through either CUDA or HIP API compatibility layers.

## Reference Headers
- **CUDA API Reference**: `/home/ben/src/iree-build/build_tools/third_party/cuda/12.2.1/linux-x86_64/include/cuda.h`
- **HIP API Reference**: `/home/ben/src/iree/third_party/hip-build-deps/include/hip/hip_runtime_api.h`

## Architecture

### Three-Layer Design

1. **API Layer** (cuda_api.c, hip_api.c)
  - Exports official CUDA/HIP APIs
  - Converts between platform error codes and iree_status_t
  - Marshals platform types to/from Stream HAL types

2. **Stream HAL Layer** (stream_*.c)
  - Platform-agnostic implementation
  - All functions return iree_status_t
  - Uses generic types (no CUDA/HIP specific types or enums)
  - Contains all business logic

3. **IREE HAL Layer**
  - Existing IREE HAL interfaces
  - May require new functionality

## File Structure

```
experimental/streaming/
├── internal.h              # Internal types and function declarations
├── init.c                  # Initialization and global state management
├── context.c               # Context management
├── device.c                # Device enumeration and properties
├── event.c                 # Event synchronization primitives
├── graph.c                 # Graph construction and instantiation
├── memory.c                # Memory allocation and transfers
├── mem_pool.c              # Memory pools and async allocations
├── module.c                # Module/kernel loading and management
├── peer.c                  # Peer-to-peer operations
├── stream.c                # Stream/queue operations
├── util/buffer_table.h/.c  # void* <-> iree_hal_streaming_buffer_t mapping
├── binding/cuda/api.h      # CUDA type definitions (matching official cuda.h)
├── binding/cuda/api.c      # CUDA API implementation
├── binding/hip/api.h       # HIP type definitions (matching hip_runtime_api.h)
└── binding/hip/api.c       # HIP API implementation
```

## Naming Conventions

### Internal Functions
- Prefix: `iree_hal_streaming_`
- Examples:
  - `iree_hal_streaming_init_global()`
  - `iree_hal_streaming_context_create()`
  - `iree_hal_streaming_buffer_wrap()`

### Internal Types
- Suffix: `_t`
- Examples:
  - `iree_hal_streaming_context_t`
  - `iree_hal_streaming_device_entry_t`
  - `iree_hal_streaming_stream_t`

### File Names
- Short name of the operation grouping
- Examples:
  - `context.c` (context creation and management)
  - `graph.c` (graph recording and instantiation)

## Error Code Handling

### Pattern
1. Internal functions return `iree_status_t`
2. API layers convert to platform-specific error codes
3. Use exact values from official headers

### CUDA Error Conversion
```c
// In cuda_api.c
CUresult iree_hal_streaming_status_to_cuda_result(iree_status_t status);
iree_status_t iree_hal_streaming_cuda_result_to_status(CUresult result);
```

### HIP Error Conversion
```c
// In hip_api.c
hipError_t iree_hal_streaming_status_to_hip_error(iree_status_t status);
iree_status_t iree_hal_streaming_hip_error_to_status(hipError_t error);
```

## Implementation Pattern

### Internal Function (returns iree_status_t)
```c
// In stream_init.c:
iree_status_t iree_hal_streaming_init_global(unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Implementation...
  iree_status_t status = ...;
  IREE_TRACE_ZONE_END(z0);
  return status;
}
```

### CUDA API Function
```c
// In cuda_api.c:
CUDAAPI CUresult cuInit(unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_streaming_init_global(Flags);
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}
```

### HIP API Function
```c
// In hip_api.c
HIPAPI hipError_t hipInit(unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_streaming_init_global(flags);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}
```

## Migration Steps

### Phase 4: Testing & Validation
1. **Compilation Testing**
   - Build both CUDA and HIP API libraries
   - Verify no compilation errors or warnings
   - Test with different compiler flags and optimization levels

2. **API Compatibility Testing**
   - Verify CUDA API compatibility using official CUDA samples
   - Verify HIP API compatibility using official HIP samples
   - Test API function signatures match official headers
   - Validate error code mappings are correct

3. **Functional Testing**
   - Basic initialization and cleanup
   - Device enumeration and properties
   - Context creation and management
   - Memory allocation and transfers
   - Kernel loading and execution
   - Stream synchronization
   - Event recording and timing

4. **Error Handling Testing**
   - Test thread-local error state management
   - Verify error propagation from internal Stream HAL
   - Test error string functions return correct messages
   - Validate sticky vs non-sticky error behavior

5. **Performance Testing**
   - Memory bandwidth benchmarks
   - Kernel launch overhead measurements
   - Stream synchronization latency
   - Compare performance with native CUDA/HIP

6. **Integration Testing**
   - Test with IREE HAL integration
   - Verify tracing works correctly
   - Test multi-device scenarios
   - P2P memory access validation

7. **Stress Testing**
   - Memory leak detection
   - Thread safety validation
   - Resource exhaustion scenarios
   - Long-running stability tests

## Build Configuration

## Benefits

1. **Code Reuse**: Single implementation for both APIs
2. **Maintainability**: Bug fixes benefit both APIs
3. **Extensibility**: Easy to add new APIs (OpenCL, etc.)
4. **Testing**: Single test suite validates both APIs
5. **Tracing**: Consistent tracing across all layers
6. **Future-Proof**: Can support non-GPU stream-based accelerators

## Pending Cleanup

- [ ] Handle statuses better
- [ ] Proper image size detection (use fatbin util, elf util, etc)
  - Add executable create flag for ("unsafe unknown image size")
  - Take flag on create_from_memory/create_from_file and OR in fields
- [ ] Symbol metadata queries in HAL
- [ ] Parameter packing
- [ ] Split internal.h into one file per logical group (matching .c)
  - [ ] May need a `common.h`-like header to break cycles
- [ ] Add extension methods (like cuHAL... and hipHAL...) for HAL type access
  - Map a pointer to an iree_hal_buffer_t* for interop
  - Get an iree_hal_device_t* from a CUDA/HIP device
  - Maybe others? HAL APIs for import/export are best for most things, but
    toll-free access would be useful when tightly interoping

## Unsupported Features

### **Graph Execution**
**Status**: Recording supported, execution not implemented
- **Location**: `graph.c:451` - `iree_hal_streaming_graph_instantiate()`
- **Impact**: Graph nodes are not converted to command buffers for execution
- **Affected CUDA APIs**:
  - `cuGraphInstantiate` / `cuGraphInstantiateWithFlags` - Returns success but doesn't create executable
  - `cuGraphLaunch` - May not execute correctly
  - `cuGraphExecUpdate` - Returns UNIMPLEMENTED
- **Affected HIP APIs**:
  - `hipGraphInstantiate` / `hipGraphInstantiateWithParams`
  - `hipGraphLaunch`
  - `hipGraphExecUpdate`

### **Cooperative Kernels**
**Status**: Not supported
- **Location**: `stream.c:344-349` - Returns IREE_STATUS_UNIMPLEMENTED
- **Impact**: Cannot launch kernels with grid synchronization
- **Affected CUDA APIs**:
  - `cuLaunchCooperativeKernel` - Would fail with UNIMPLEMENTED
  - `cuLaunchCooperativeKernelMultiDevice`
- **Affected HIP APIs**:
  - `hipLaunchCooperativeKernel`
  - `hipLaunchCooperativeKernelMultiDevice`

### **Managed/Unified Memory**
**Status**: Not implemented
- **Location**: `cuda_api.c:1428`, `hip_api.c:2680`
- **Impact**: Cannot use unified memory programming model
- **Affected CUDA APIs**:
  - `cuMemAllocManaged` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuMemAdvise` - TODO placeholder, no-op
  - `cuMemPrefetchAsync` - TODO placeholder, no-op
  - Various `cuMemRange*` functions for managed memory attributes
- **Affected HIP APIs**:
  - `hipMallocManaged` - Returns error
  - `hipMemAdvise` - TODO placeholder
  - `hipMemPrefetchAsync` - TODO placeholder
  - `hipMemRangeGetAttribute` / `hipMemRangeGetAttributes`

### **IPC (Inter-Process Communication)**
**Status**: Not supported
- **Location**: Multiple locations in cuda_api.c
- **Impact**: Cannot share memory or events between processes
- **Affected CUDA APIs**:
  - `cuIpcGetEventHandle` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuIpcOpenEventHandle` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuIpcGetMemHandle` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuIpcOpenMemHandle` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuIpcCloseMemHandle` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuMemPoolExportToShareableHandle` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuMemPoolImportFromShareableHandle` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuMemPoolExportPointer` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuMemPoolImportPointer` - Returns CUDA_ERROR_NOT_SUPPORTED
- **Affected HIP APIs**:
  - `hipIpcGetMemHandle`
  - `hipIpcOpenMemHandle`
  - `hipIpcCloseMemHandle`
  - `hipIpcGetEventHandle`
  - `hipIpcOpenEventHandle`

### **Async Memory Pool Operations**
**Status**: Partially implemented
- **Location**: `mem_pool.c:241-259`
- **Impact**: Cannot allocate/free memory asynchronously from pools
- **Affected CUDA APIs**:
  - `cuMemAllocFromPoolAsync` - Would fail when pool operations are async
  - `cuMemFreeAsync` - Returns UNIMPLEMENTED for actual async operations
- **Affected HIP APIs**:
  - `hipMallocFromPoolAsync`
  - `hipFreeAsync`

### **PCI Bus ID Operations**
**Status**: Not implemented
- **Location**: `cuda_api.c:1010-1019`
- **Impact**: Cannot query devices by PCI bus ID
- **Affected CUDA APIs**:
  - `cuDeviceGetByPCIBusId` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuDeviceGetPCIBusId` - Returns CUDA_ERROR_NOT_SUPPORTED
- **Affected HIP APIs**:
  - `hipDeviceGetByPCIBusId`
  - `hipDeviceGetPCIBusId`

### **Memory Pool Access Control**
**Status**: Not implemented
- **Location**: `cuda_api.c:3786-3798`
- **Impact**: Cannot set fine-grained memory pool access permissions
- **Affected CUDA APIs**:
  - `cuMemPoolSetAccess` - Returns CUDA_ERROR_NOT_SUPPORTED
  - `cuMemPoolGetAccess` - Returns CUDA_ERROR_NOT_SUPPORTED
- **Affected HIP APIs**:
  - `hipMemPoolSetAccess`
  - `hipMemPoolGetAccess`

### **P2P Capabilities**
**Status**: Hardcoded/not queried from driver
- **Location**: `init.c:241`, `context.c:466`
- **Impact**: P2P support is assumed but not validated
- **TODOs**: Query actual P2P capabilities from driver

### **Device Properties**
**Status**: Hardcoded values
- **Location**: `init.c:53-63`
- **Impact**: Returns placeholder values instead of actual device properties
- **TODOs**: Query from actual device properties

### **Module Metadata**
**Status**: Not parsed
- **Location**: `module.c:45`
- **Impact**: Cannot extract symbol metadata from binaries
- **TODO**: Parse actual symbol metadata from binary format

### **Memory Pool Trimming**
**Status**: Not implemented
- **Location**: `mem_pool.c:168`
- **Impact**: Cannot reclaim unused memory from pools
- **TODO**: Implement memory pool trimming

### **Event timestamps**
**Status**: Not supported
- **Location**: `event.c`
- **Impact**: Cannot time execution on device
- **TODO**: Add programmatic timestamps to the HAL API and use (if possible)

### **Not Implemented API Categories** (No code present)
- **Textures/Surfaces**: No texture reference or surface APIs
- **Arrays**: No CUDA array support (`cuArray*` functions)
- **Virtual Memory Management**: No `cuMemAddressReserve`, `cuMemMap`, etc.
- **External Memory**: No external memory import/export
- **Graphics Interop**: No OpenGL/DirectX interoperability
- **CUDA Contexts**: Limited context API support (maybe? what?)

## Enhancements

### Flag to kernel launch for using the IREE HAL ABI

Today we disable a lot of IREE functionality to support custom kernel arguments.
If the user is following the HAL ABI and we know that either on the executable
or on the launch we can make assumptions about the kernel argument format passed
to cuLaunchKernel, including turning void* buffer references into
iree_hal_buffer_t* references we can track. This would avoid the need for user
code to pack arguments differently or provide different sources for their
kernels. It could be a symbol flag that we either automatically detect (when
possible per HAL target) or let the user override with hipFuncSetAttribute etc.

### Use iree_status_t for all errors and only adapt at the end

Instead of this, which has several return points and ignores the IREE status with useful information:
```c
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phGraph) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Get current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Look up the buffer containing this range.
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_hal_streaming_memory_lookup_range(context, devPtr, count, &buffer);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }
```

We should have something like this:
```c
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phGraph) {
    IREE_RETURN_CU_RESULT_AND_END_ZONE_IF_ERROR(z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "phGraph must not be NULL"));
  }

  // Get current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_RETURN_CU_RESULT_AND_END_ZONE_IF_ERROR(z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no context assigned to the calling thread"));
  }

  // Look up the buffer containing this range.
  iree_hal_streaming_buffer_t* buffer = NULL;
  // IREE_RETURN_CU_RESULT_AND_END_ZONE_IF_ERROR is like
  // IREE_RETURN_AND_END_ZONE_IF_ERROR but takes an optional CUDA result that
  // overrides the status code of the provided iree_status_t. This is required
  // for matching the CUDA API (our internal status code may not always map to
  // what the user may be expecting).
  //
  // A matching IREE_RETURN_CU_RESULT_IF_ERROR without the tracing zone would
  // be useful for functions that don't use tracing.
  IREE_RETURN_CU_RESULT_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup_range(context, devPtr, count, &buffer),
      CUDA_ERROR_INVALID_VALUE);

  ...
```

There's probably some other macros we could use too. For the unconditional returns it'd be nice to have a non-IF_ERROR version to save some characters. If there's a lot of repetition of the same error types (NULL checks, context checks, etc) we could have macros just for those (`IREE_RETURN_CUDA_ERROR_INVALID_VALUE("phGraph")`) that would expand to the other macros.

The end goal would be to have a logging mode that when enabled logs out all the status errors passed to `IREE_RETURN_CU_RESULT_*` before returning. We could also change from using the tracy begin/end zone macros to our own that allows for logging the API call.

Consistently using these macros in the CUDA API and then updating the existing HIP macros (which track thread-local sticky error state already) to match would make it easier to switch between the files making changes as the error handling would be consistent.

## Implementation Priority

**High Priority** (Core functionality):
- [ ] Module metadata parsing - Better kernel introspection
- [ ] Graph execution - Critical for command buffer optimization
- [ ] Async memory operations - Required for performance
- [ ] Device property queries - Needed for correct behavior
- [ ] Event timestamps - Heavily used (incorrectly) in user benchmark programs

**Medium Priority** (Important features):
- [ ] Managed/unified memory - Simplifies programming model
- [ ] P2P capabilities - Multi-GPU support

**Low Priority** (Advanced/specialized):
- [ ] IPC support - Only needed for multi-process scenarios
- [ ] Cooperative kernels - Specialized use case
- [ ] PCI Bus ID - Specific deployment scenarios
- [ ] Memory pool access control - Fine-grained permissions

