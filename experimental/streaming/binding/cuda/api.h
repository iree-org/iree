// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_BINDING_CUDA_API_H_
#define IREE_EXPERIMENTAL_STREAMING_BINDING_CUDA_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Export macros
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#ifdef IREE_HAL_STREAMING_CUDA_EXPORTS
#define CUDAAPI __declspec(dllexport)
#else
#define CUDAAPI __declspec(dllimport)
#endif  // IREE_HAL_STREAMING_CUDA_EXPORTS
#else
#define CUDAAPI __attribute__((visibility("default")))
#endif  // _WIN32

//===----------------------------------------------------------------------===//
// CUDA API types and definitions
//===----------------------------------------------------------------------===//

typedef enum cudaError_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_DEINITIALIZED = 4,
  CUDA_ERROR_PROFILER_DISABLED = 5,
  CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
  CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
  CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
  CUDA_ERROR_STUB_LIBRARY = 34,
  CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
  CUDA_ERROR_NO_DEVICE = 100,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
  CUDA_ERROR_INVALID_IMAGE = 200,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
  CUDA_ERROR_MAP_FAILED = 205,
  CUDA_ERROR_UNMAP_FAILED = 206,
  CUDA_ERROR_ARRAY_IS_MAPPED = 207,
  CUDA_ERROR_ALREADY_MAPPED = 208,
  CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
  CUDA_ERROR_ALREADY_ACQUIRED = 210,
  CUDA_ERROR_NOT_MAPPED = 211,
  CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
  CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
  CUDA_ERROR_ECC_UNCORRECTABLE = 214,
  CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
  CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
  CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
  CUDA_ERROR_INVALID_PTX = 218,
  CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
  CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
  CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
  CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
  CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
  CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
  CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
  CUDA_ERROR_INVALID_SOURCE = 300,
  CUDA_ERROR_FILE_NOT_FOUND = 301,
  CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
  CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
  CUDA_ERROR_OPERATING_SYSTEM = 304,
  CUDA_ERROR_INVALID_HANDLE = 400,
  CUDA_ERROR_ILLEGAL_STATE = 401,
  CUDA_ERROR_NOT_FOUND = 500,
  CUDA_ERROR_NOT_READY = 600,
  CUDA_ERROR_ILLEGAL_ADDRESS = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT = 702,
  CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
  CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
  CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
  CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
  CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
  CUDA_ERROR_ASSERT = 710,
  CUDA_ERROR_TOO_MANY_PEERS = 711,
  CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
  CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
  CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
  CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
  CUDA_ERROR_MISALIGNED_ADDRESS = 716,
  CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
  CUDA_ERROR_INVALID_PC = 718,
  CUDA_ERROR_LAUNCH_FAILED = 719,
  CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
  CUDA_ERROR_NOT_PERMITTED = 800,
  CUDA_ERROR_NOT_SUPPORTED = 801,
  CUDA_ERROR_SYSTEM_NOT_READY = 802,
  CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
  CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
  CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
  CUDA_ERROR_MPS_RPC_FAILURE = 806,
  CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
  CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
  CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
  CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
  CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
  CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
  CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
  CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
  CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
  CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
  CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
  CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
  CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
  CUDA_ERROR_CAPTURED_EVENT = 907,
  CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
  CUDA_ERROR_TIMEOUT = 909,
  CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
  CUDA_ERROR_EXTERNAL_DEVICE = 911,
  CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
  CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef struct CUarray_st* CUarray;
typedef int CUdevice;
typedef uintptr_t CUdeviceptr;

typedef enum CUmemorytype {
  CU_MEMORYTYPE_HOST = 0x01,
  CU_MEMORYTYPE_DEVICE = 0x02,
  CU_MEMORYTYPE_ARRAY = 0x03,
  CU_MEMORYTYPE_UNIFIED = 0x04,
} CUmemorytype;

#define CU_CTX_SCHED_AUTO 0x00
#define CU_CTX_SCHED_SPIN 0x01
#define CU_CTX_SCHED_YIELD 0x02
#define CU_CTX_SCHED_BLOCKING_SYNC 0x04
#define CU_CTX_SCHED_MASK 0x07  // Mask for scheduling mode bits
#define CU_CTX_MAP_HOST 0x08
#define CU_CTX_LMEM_RESIZE_TO_MAX 0x10

typedef enum CUlimit_enum {
  CU_LIMIT_STACK_SIZE = 0,
  CU_LIMIT_PRINTF_FIFO_SIZE = 1,
  CU_LIMIT_MALLOC_HEAP_SIZE = 2,
  CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
  CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
  CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
  CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6,
  CU_LIMIT_MAX
} CUlimit;

#define CU_STREAM_DEFAULT 0x0
#define CU_STREAM_NON_BLOCKING 0x1

#define CU_EVENT_DEFAULT 0x0
#define CU_EVENT_BLOCKING_SYNC 0x1
#define CU_EVENT_DISABLE_TIMING 0x2
#define CU_EVENT_INTERPROCESS 0x4

#define CU_MEMHOSTALLOC_PORTABLE 0x01
#define CU_MEMHOSTALLOC_DEVICEMAP 0x02
#define CU_MEMHOSTALLOC_WRITECOMBINED 0x04

typedef enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
  CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
  CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
  CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
  CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
  CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
  CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
  CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
  CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
  CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
  CU_DEVICE_ATTRIBUTE_MAX = 135
} CUdevice_attribute;

typedef enum CUdevice_P2PAttribute_enum {
  CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1,
  CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2,
  CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3,
  CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4,
  CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 8
} CUdevice_P2PAttribute;

typedef enum CUfunction_attribute_enum {
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
  CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
  CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
  CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
  CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
  CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;

typedef enum CUfunc_cache {
  CU_FUNC_CACHE_PREFER_NONE = 0,
  CU_FUNC_CACHE_PREFER_SHARED = 1,
  CU_FUNC_CACHE_PREFER_L1 = 2,
  CU_FUNC_CACHE_PREFER_EQUAL = 3,
} CUfunc_cache;

typedef enum CUsharedconfig {
  CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0,
  CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1,
  CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2,
} CUsharedconfig;

typedef enum CUjit_option_enum {
  CU_JIT_MAX_REGISTERS = 0,
  CU_JIT_THREADS_PER_BLOCK = 1,
  CU_JIT_WALL_TIME = 2,
  CU_JIT_INFO_LOG_BUFFER = 3,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
  CU_JIT_ERROR_LOG_BUFFER = 5,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
  CU_JIT_OPTIMIZATION_LEVEL = 7,
  CU_JIT_TARGET_FROM_CUCONTEXT = 8,
  CU_JIT_TARGET = 9,
  CU_JIT_FALLBACK_STRATEGY = 10,
  CU_JIT_GENERATE_DEBUG_INFO = 11,
  CU_JIT_LOG_VERBOSE = 12,
  CU_JIT_GENERATE_LINE_INFO = 13,
  CU_JIT_CACHE_MODE = 14,
  CU_JIT_NEW_SM3X_OPT = 15,
  CU_JIT_FAST_COMPILE = 16,
  CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
  CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
  CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
  CU_JIT_NUM_OPTIONS
} CUjit_option;

typedef struct CUDA_LAUNCH_PARAMS {
  CUfunction function;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  void** kernelParams;
} CUDA_LAUNCH_PARAMS;

typedef struct CUuuid {
  unsigned char bytes[16];
} CUuuid;

typedef struct CUipcEventHandle_st {
  char reserved[64];
} CUipcEventHandle;

typedef struct CUipcMemHandle_st {
  char reserved[64];
} CUipcMemHandle;

typedef void (*CUhostFn)(void* userData);

typedef size_t (*CUoccupancyB2DSize)(int blockSize);

// Host memory registration flags.
#define CU_MEMHOSTREGISTER_PORTABLE 0x01
#define CU_MEMHOSTREGISTER_DEVICEMAP 0x02
#define CU_MEMHOSTREGISTER_IOMEMORY 0x04
#define CU_MEMHOSTREGISTER_READ_ONLY 0x08

// Graph types.
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphExec_st* CUgraphExec;
typedef struct CUgraphNode_st* CUgraphNode;

// Memory advice enum.
typedef enum CUmem_advise_enum {
  CU_MEM_ADVISE_SET_READ_MOSTLY = 1,
  CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
  CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
  CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
  CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
  CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6,
} CUmem_advise;

// Pointer attribute enum.
typedef enum CUpointer_attribute_enum {
  CU_POINTER_ATTRIBUTE_CONTEXT = 1,
  CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
  CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
  CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
  CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
  CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
  CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
  CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
  CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
  CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
  CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
  CU_POINTER_ATTRIBUTE_MAPPED = 13,
  CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
  CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
  CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
  CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
} CUpointer_attribute;

// Memory range attribute enum.
typedef enum CUmem_range_attribute_enum {
  CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
  CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
  CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
  CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
} CUmem_range_attribute;

// Stream capture mode.
typedef enum CUstreamCaptureMode_enum {
  CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
  CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
  CU_STREAM_CAPTURE_MODE_RELAXED = 2,
} CUstreamCaptureMode;

// Stream capture status.
typedef enum CUstreamCaptureStatus_enum {
  CU_STREAM_CAPTURE_STATUS_NONE = 0,
  CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
  CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
} CUstreamCaptureStatus;

// Graph node type.
typedef enum CUgraphNodeType_enum {
  CU_GRAPH_NODE_TYPE_KERNEL = 0,
  CU_GRAPH_NODE_TYPE_MEMCPY = 1,
  CU_GRAPH_NODE_TYPE_MEMSET = 2,
  CU_GRAPH_NODE_TYPE_HOST = 3,
  CU_GRAPH_NODE_TYPE_GRAPH = 4,
  CU_GRAPH_NODE_TYPE_EMPTY = 5,
  CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
  CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
  CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
  CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
  CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
  CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
} CUgraphNodeType;

// Graph instantiate flags.
typedef enum CUgraphInstantiate_flags_enum {
  CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
  CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2,
  CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4,
  CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
} CUgraphInstantiate_flags;

// Stream capture dependency update flags.
#define CU_STREAM_ADD_CAPTURE_DEPENDENCIES 0x1
#define CU_STREAM_SET_CAPTURE_DEPENDENCIES 0x2

// Kernel launch parameter markers.
#define CU_LAUNCH_PARAM_END ((void*)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)

//===----------------------------------------------------------------------===//
// Graph node parameter structures
//===----------------------------------------------------------------------===//

// Kernel node parameters.
typedef struct CUDA_KERNEL_NODE_PARAMS_st {
  CUfunction func;              // Kernel function.
  unsigned int gridDimX;        // Grid X dimension.
  unsigned int gridDimY;        // Grid Y dimension.
  unsigned int gridDimZ;        // Grid Z dimension.
  unsigned int blockDimX;       // Block X dimension.
  unsigned int blockDimY;       // Block Y dimension.
  unsigned int blockDimZ;       // Block Z dimension.
  unsigned int sharedMemBytes;  // Dynamic shared memory size.
  void** kernelParams;          // Array of kernel parameters.
  void** extra;                 // Extra options.
} CUDA_KERNEL_NODE_PARAMS;

// Memory copy node parameters.
typedef struct CUDA_MEMCPY3D_st {
  size_t srcXInBytes;          // Source X in bytes.
  size_t srcY;                 // Source Y.
  size_t srcZ;                 // Source Z.
  size_t srcLOD;               // Source LOD.
  CUmemorytype srcMemoryType;  // Source memory type.
  const void* srcHost;         // Source host pointer.
  CUdeviceptr srcDevice;       // Source device pointer.
  CUarray srcArray;            // Source array.
  void* reserved0;             // Reserved, must be NULL.
  size_t srcPitch;             // Source pitch.
  size_t srcHeight;            // Source height.

  size_t dstXInBytes;          // Destination X in bytes.
  size_t dstY;                 // Destination Y.
  size_t dstZ;                 // Destination Z.
  size_t dstLOD;               // Destination LOD.
  CUmemorytype dstMemoryType;  // Destination memory type.
  void* dstHost;               // Destination host pointer.
  CUdeviceptr dstDevice;       // Destination device pointer.
  CUarray dstArray;            // Destination array.
  void* reserved1;             // Reserved, must be NULL.
  size_t dstPitch;             // Destination pitch.
  size_t dstHeight;            // Destination height.

  size_t WidthInBytes;  // Width in bytes.
  size_t Height;        // Height.
  size_t Depth;         // Depth.
} CUDA_MEMCPY3D;

// Memset node parameters.
typedef struct CUDA_MEMSET_NODE_PARAMS_st {
  CUdeviceptr dst;           // Destination device pointer.
  size_t pitch;              // Pitch (0 for 1D).
  unsigned int value;        // Value to set.
  unsigned int elementSize;  // Size of each element (1, 2, or 4 bytes).
  size_t width;              // Width in elements.
  size_t height;             // Height in elements.
} CUDA_MEMSET_NODE_PARAMS;

// Host node parameters.
typedef struct CUDA_HOST_NODE_PARAMS_st {
  CUhostFn fn;     // Host function to call.
  void* userData;  // User data for the function.
} CUDA_HOST_NODE_PARAMS;

// Memory pool handle type.
typedef struct CUmemPoolHandle_st* CUmemoryPool;

// Memory allocation handle types.
typedef enum CUmemAllocationHandleType_enum {
  CU_MEM_HANDLE_TYPE_NONE = 0,
  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1,
  CU_MEM_HANDLE_TYPE_WIN32 = 2,
  CU_MEM_HANDLE_TYPE_WIN32_KMT = 4,
  CU_MEM_HANDLE_TYPE_MAX = 0x7FFFFFFF
} CUmemAllocationHandleType;

// Memory allocation types.
typedef enum CUmemAllocationType_enum {
  CU_MEM_ALLOCATION_TYPE_INVALID = 0,
  CU_MEM_ALLOCATION_TYPE_PINNED = 1,
  CU_MEM_ALLOCATION_TYPE_MAX = 0x7FFFFFFF
} CUmemAllocationType;

// Memory location types.
typedef enum CUmemLocationType_enum {
  CU_MEM_LOCATION_TYPE_INVALID = 0,
  CU_MEM_LOCATION_TYPE_DEVICE = 1,
  CU_MEM_LOCATION_TYPE_HOST = 2,
  CU_MEM_LOCATION_TYPE_HOST_NUMA = 3,
  CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 4,
  CU_MEM_LOCATION_TYPE_MAX = 0x7FFFFFFF
} CUmemLocationType;

// Memory location descriptor.
typedef struct CUmemLocation_st {
  CUmemLocationType type;
  int id;
} CUmemLocation;

// Memory pool properties.
typedef struct CUmemPoolProps_st {
  CUmemAllocationType allocType;
  CUmemAllocationHandleType handleTypes;
  CUmemLocation location;
  void* win32SecurityAttributes;
  size_t maxSize;
  unsigned char reserved[16];
} CUmemPoolProps;

// Memory pool pointer export data.
typedef struct CUmemPoolPtrExportData_st {
  unsigned char reserved[64];
} CUmemPoolPtrExportData;

// Memory pool attributes.
typedef enum CUmemPool_attribute_enum {
  CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
  CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
  CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
  CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4,
  CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5,
  CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6,
  CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7,
  CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8,
} CUmemPool_attribute;

// Memory access flags.
typedef enum CUmemAccess_flags_enum {
  CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
  CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
  CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
  CU_MEM_ACCESS_FLAGS_PROT_MAX = 0x7FFFFFFF
} CUmemAccess_flags;

// Memory access descriptor.
typedef struct CUmemAccessDesc_st {
  CUmemLocation location;
  CUmemAccess_flags flags;
} CUmemAccessDesc;

//===----------------------------------------------------------------------===//
// CUDA Driver API function declarations
//===----------------------------------------------------------------------===//

// Initialization
CUDAAPI CUresult cuInit(unsigned int Flags);
CUDAAPI CUresult cuHALDeinit(void);  // HAL extension
CUDAAPI CUresult cuDriverGetVersion(int* driverVersion);

// Device management
CUDAAPI CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUDAAPI CUresult cuDeviceGetCount(int* count);
CUDAAPI CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUDAAPI CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev);
CUDAAPI CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);
CUDAAPI CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib,
                                      CUdevice dev);
CUDAAPI CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev,
                                       CUdevice peerDev);
CUDAAPI CUresult cuDeviceGetP2PAttribute(int* value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice);
CUDAAPI CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId);
CUDAAPI CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev);

// Primary context management
CUDAAPI CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUDAAPI CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUDAAPI CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUDAAPI CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags,
                                            int* active);
CUDAAPI CUresult cuDevicePrimaryCtxReset(CUdevice dev);

// Context management
CUDAAPI CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUDAAPI CUresult cuCtxDestroy(CUcontext ctx);
CUDAAPI CUresult cuCtxPushCurrent(CUcontext ctx);
CUDAAPI CUresult cuCtxPopCurrent(CUcontext* pctx);
CUDAAPI CUresult cuCtxSetCurrent(CUcontext ctx);
CUDAAPI CUresult cuCtxGetCurrent(CUcontext* pctx);
CUDAAPI CUresult cuCtxGetDevice(CUdevice* device);
CUDAAPI CUresult cuCtxGetFlags(unsigned int* flags);
CUDAAPI CUresult cuCtxSynchronize(void);
CUDAAPI CUresult cuCtxSetLimit(CUlimit limit, size_t value);
CUDAAPI CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit);
CUDAAPI CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version);
CUDAAPI CUresult cuCtxGetStreamPriorityRange(int* leastPriority,
                                             int* greatestPriority);
CUDAAPI CUresult cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags);
CUDAAPI CUresult cuCtxDisablePeerAccess(CUcontext peerContext);

// Module management
CUDAAPI CUresult cuModuleLoad(CUmodule* module, const char* fname);
CUDAAPI CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUDAAPI CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                                    unsigned int numOptions,
                                    CUjit_option* options, void** optionValues);
CUDAAPI CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin);
CUDAAPI CUresult cuModuleUnload(CUmodule hmod);
CUDAAPI CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                                     const char* name);
CUDAAPI CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                                   CUmodule hmod, const char* name);

// Memory management
CUDAAPI CUresult cuMemGetInfo(size_t* free, size_t* total);
CUDAAPI CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUDAAPI CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes);
CUDAAPI CUresult cuMemFree(CUdeviceptr dptr);
CUDAAPI CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize,
                                      CUdeviceptr dptr);
CUDAAPI CUresult cuMemAllocHost(void** pp, size_t bytesize);
CUDAAPI CUresult cuMemFreeHost(void* p);
CUDAAPI CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
CUDAAPI CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p,
                                           unsigned int Flags);
CUDAAPI CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p);
CUDAAPI CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize,
                                   unsigned int flags);
CUDAAPI CUresult cuMemHostRegister(void* p, size_t bytesize,
                                   unsigned int Flags);
CUDAAPI CUresult cuMemHostUnregister(void* p);

// Synchronous memory copies
CUDAAPI CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUDAAPI CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount);
CUDAAPI CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost,
                              size_t ByteCount);
CUDAAPI CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount);
CUDAAPI CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount);

// Asynchronous memory copies
CUDAAPI CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream);
CUDAAPI CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream);
CUDAAPI CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                                   size_t ByteCount, CUstream hStream);
CUDAAPI CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream);
CUDAAPI CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream);

// Memory set
CUDAAPI CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUDAAPI CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us,
                             size_t N);
CUDAAPI CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUDAAPI CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream);
CUDAAPI CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream);
CUDAAPI CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream);

// IPC memory handles
CUDAAPI CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr);
CUDAAPI CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle,
                                    unsigned int Flags);
CUDAAPI CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);

// Memory pool management
CUDAAPI CUresult cuMemPoolCreate(CUmemoryPool* pool,
                                 const CUmemPoolProps* poolProps);
CUDAAPI CUresult cuMemPoolDestroy(CUmemoryPool pool);
CUDAAPI CUresult cuMemPoolSetAttribute(CUmemoryPool pool,
                                       CUmemPool_attribute attr, void* value);
CUDAAPI CUresult cuMemPoolGetAttribute(CUmemoryPool pool,
                                       CUmemPool_attribute attr, void* value);
CUDAAPI CUresult cuMemPoolSetAccess(CUmemoryPool pool,
                                    const CUmemAccessDesc* map, size_t count);
CUDAAPI CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool pool,
                                    CUmemLocation* location);
CUDAAPI CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
CUDAAPI CUresult cuMemPoolExportToShareableHandle(
    void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType,
    unsigned long long flags);
CUDAAPI CUresult cuMemPoolImportFromShareableHandle(
    CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType,
    unsigned long long flags);
CUDAAPI CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out,
                                        CUdeviceptr ptr);
CUDAAPI CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool,
                                        CUmemPoolPtrExportData* shareData);

// Device memory pool management
CUDAAPI CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);
CUDAAPI CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev);
CUDAAPI CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out,
                                           CUdevice dev);

// Async memory allocation
CUDAAPI CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize,
                                 CUstream hStream);
CUDAAPI CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize,
                                         CUmemoryPool pool, CUstream hStream);
CUDAAPI CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);

// Function management
CUDAAPI CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib,
                                    CUfunction hfunc);
CUDAAPI CUresult cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value);
CUDAAPI CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
CUDAAPI CUresult cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config);

// Kernel execution
CUDAAPI CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void** kernelParams, void** extra);
CUDAAPI CUresult cuLaunchCooperativeKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void** kernelParams);
CUDAAPI CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn,
                                  void* userData);

// Occupancy
CUDAAPI CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
CUDAAPI CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags);
CUDAAPI CUresult cuOccupancyMaxPotentialBlockSize(
    int* minGridSize, int* blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit);

// Event management
CUDAAPI CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags);
CUDAAPI CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUDAAPI CUresult cuEventQuery(CUevent hEvent);
CUDAAPI CUresult cuEventSynchronize(CUevent hEvent);
CUDAAPI CUresult cuEventDestroy(CUevent hEvent);
CUDAAPI CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart,
                                    CUevent hEnd);
CUDAAPI CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event);
CUDAAPI CUresult cuIpcOpenEventHandle(CUevent* phEvent,
                                      CUipcEventHandle handle);

// Stream management
CUDAAPI CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags);
CUDAAPI CUresult cuStreamCreateWithPriority(CUstream* phStream,
                                            unsigned int flags, int priority);
CUDAAPI CUresult cuStreamGetPriority(CUstream hStream, int* priority);
CUDAAPI CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags);
CUDAAPI CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx);
CUDAAPI CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags);
CUDAAPI CUresult cuStreamQuery(CUstream hStream);
CUDAAPI CUresult cuStreamSynchronize(CUstream hStream);
CUDAAPI CUresult cuStreamDestroy(CUstream hStream);
CUDAAPI CUresult cuStreamCopyAttributes(CUstream dst, CUstream src);

// Unified memory management
CUDAAPI CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device);
CUDAAPI CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream);
CUDAAPI CUresult cuPointerGetAttribute(void* data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr);
CUDAAPI CUresult cuPointerSetAttribute(const void* value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr);
CUDAAPI CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute* attributes,
                                        void** data, CUdeviceptr ptr);
CUDAAPI CUresult cuMemRangeGetAttribute(void* data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count);
CUDAAPI CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes,
                                         CUmem_range_attribute* attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count);

// CUDA graphs
CUDAAPI CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags);
CUDAAPI CUresult cuGraphDestroy(CUgraph hGraph);
CUDAAPI CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph,
                                    CUgraphNode* phErrorNode, char* logBuffer,
                                    size_t bufferSize);
CUDAAPI CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec,
                                             CUgraph hGraph,
                                             unsigned long long flags);
CUDAAPI CUresult cuGraphExecDestroy(CUgraphExec hGraphExec);
CUDAAPI CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
CUDAAPI CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                                   CUgraphNode* hErrorNode_out,
                                   unsigned int flags);
CUDAAPI CUresult cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                      const CUgraphNode* dependencies,
                                      size_t numDependencies,
                                      const void* nodeParams);
CUDAAPI CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                      const CUgraphNode* dependencies,
                                      size_t numDependencies,
                                      const void* copyParams, CUcontext ctx);
CUDAAPI CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                      const CUgraphNode* dependencies,
                                      size_t numDependencies,
                                      const void* memsetParams, CUcontext ctx);
CUDAAPI CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                    const CUgraphNode* dependencies,
                                    size_t numDependencies,
                                    const void* hostParams);
CUDAAPI CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                     const CUgraphNode* dependencies,
                                     size_t numDependencies);

// Stream capture
CUDAAPI CUresult cuStreamBeginCapture(CUstream hStream,
                                      CUstreamCaptureMode mode);
CUDAAPI CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph);
CUDAAPI CUresult cuStreamIsCapturing(CUstream hStream,
                                     CUstreamCaptureStatus* captureStatus);
CUDAAPI CUresult cuStreamGetCaptureInfo(CUstream hStream,
                                        CUstreamCaptureStatus* captureStatus,
                                        unsigned long long* id);
CUDAAPI CUresult cuStreamUpdateCaptureDependencies(CUstream hStream,
                                                   CUgraphNode* dependencies,
                                                   size_t numDependencies,
                                                   unsigned int flags);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_BINDING_CUDA_API_H_
