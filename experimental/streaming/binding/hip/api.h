// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_BINDING_HIP_API_H_
#define IREE_EXPERIMENTAL_STREAMING_BINDING_HIP_API_H_

// HIP API compatibility layer
// This allows HIP applications to run on IREE Stream HAL backends

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Export macros
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#ifdef IREE_HAL_STREAMING_HIP_EXPORTS
#define HIPAPI __declspec(dllexport)
#else
#define HIPAPI __declspec(dllimport)
#endif  // IREE_HAL_STREAMING_HIP_EXPORTS
#else
#define HIPAPI __attribute__((visibility("default")))
#endif  // _WIN32

//===----------------------------------------------------------------------===//
// HIP types
//===----------------------------------------------------------------------===//

typedef int hipDevice_t;
typedef struct hipCtx_st* hipCtx_t;
typedef struct hipModule_st* hipModule_t;
typedef struct hipFunction_st* hipFunction_t;
typedef struct hipStream_st* hipStream_t;
typedef struct hipEvent_st* hipEvent_t;
typedef struct hipArray_st* hipArray_t;
typedef void* hipDeviceptr_t;

// Dimension type.
typedef struct dim3 {
  unsigned int x, y, z;
} dim3;

// Pitched pointer type.
typedef struct hipPitchedPtr {
  void* ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
} hipPitchedPtr;

// Context scheduling flags (matching CUDA).
#define hipDeviceScheduleAuto 0x00
#define hipDeviceScheduleSpin 0x01
#define hipDeviceScheduleYield 0x02
#define hipDeviceScheduleBlockingSync 0x04
#define hipDeviceMapHost 0x08
#define hipDeviceLmemResizeToMax 0x10
#define hipDeviceScheduleMask 0x07  // Mask for scheduling mode bits

typedef enum __attribute__((annotate("HIP_nodiscard"))) hipError_t {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorNotInitialized = 3,
  hipErrorDeinitialized = 4,
  hipErrorProfilerDisabled = 5,
  hipErrorProfilerNotInitialized = 6,
  hipErrorProfilerAlreadyStarted = 7,
  hipErrorProfilerAlreadyStopped = 8,
  hipErrorInvalidConfiguration = 9,
  hipErrorInvalidPitchValue = 12,
  hipErrorInvalidSymbol = 13,
  hipErrorInvalidDevicePointer = 17,
  hipErrorInvalidMemcpyDirection = 21,
  hipErrorInsufficientDriver = 35,
  hipErrorMissingConfiguration = 52,
  hipErrorPriorLaunchFailure = 53,
  hipErrorInvalidDeviceFunction = 98,
  hipErrorNoDevice = 100,
  hipErrorInvalidDevice = 101,
  hipErrorInvalidImage = 200,
  hipErrorInvalidContext = 201,
  hipErrorContextAlreadyCurrent = 202,
  hipErrorMapFailed = 205,
  hipErrorMapBufferObjectFailed = 205,  // Deprecated
  hipErrorUnmapFailed = 206,
  hipErrorArrayIsMapped = 207,
  hipErrorAlreadyMapped = 208,
  hipErrorNoBinaryForGpu = 209,
  hipErrorAlreadyAcquired = 210,
  hipErrorNotMapped = 211,
  hipErrorNotMappedAsArray = 212,
  hipErrorNotMappedAsPointer = 213,
  hipErrorECCNotCorrectable = 214,
  hipErrorUnsupportedLimit = 215,
  hipErrorContextAlreadyInUse = 216,
  hipErrorPeerAccessUnsupported = 217,
  hipErrorInvalidKernelFile = 218,
  hipErrorInvalidGraphicsContext = 219,
  hipErrorInvalidSource = 300,
  hipErrorFileNotFound = 301,
  hipErrorSharedObjectSymbolNotFound = 302,
  hipErrorSharedObjectInitFailed = 303,
  hipErrorOperatingSystem = 304,
  hipErrorInvalidHandle = 400,
  hipErrorInvalidResourceHandle = 400,  // Deprecated
  hipErrorIllegalState = 401,
  hipErrorNotFound = 500,
  hipErrorNotReady = 600,
  hipErrorIllegalAddress = 700,
  hipErrorLaunchOutOfResources = 701,
  hipErrorLaunchTimeOut = 702,
  hipErrorPeerAccessAlreadyEnabled = 704,
  hipErrorPeerAccessNotEnabled = 705,
  hipErrorSetOnActiveProcess = 708,
  hipErrorContextIsDestroyed = 709,
  hipErrorAssert = 710,
  hipErrorHostMemoryAlreadyRegistered = 712,
  hipErrorHostMemoryNotRegistered = 713,
  hipErrorLaunchFailure = 719,
  hipErrorCooperativeLaunchTooLarge = 720,
  hipErrorNotSupported = 801,
  hipErrorStreamCaptureUnsupported = 900,
  hipErrorStreamCaptureInvalidated = 901,
  hipErrorStreamCaptureMerge = 902,
  hipErrorStreamCaptureUnmatched = 903,
  hipErrorStreamCaptureUnjoined = 904,
  hipErrorStreamCaptureIsolation = 905,
  hipErrorStreamCaptureImplicit = 906,
  hipErrorCapturedEvent = 907,
  hipErrorStreamCaptureWrongThread = 908,
  hipErrorGraphExecUpdateFailure = 910,
  hipErrorUnknown = 999,
  hipErrorRuntimeMemory = 1052,
  hipErrorRuntimeOther = 1053,
  hipErrorTbd = 9999  // Placeholder
} hipError_t;

typedef enum hipDeviceAttribute_t {
  hipDeviceAttributeCudaCompatibleBegin = 0,
  hipDeviceAttributeEccEnabled = 0,
  hipDeviceAttributeAccessPolicyMaxWindowSize = 1,
  hipDeviceAttributeAsyncEngineCount = 2,
  hipDeviceAttributeCanMapHostMemory = 3,
  hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4,
  hipDeviceAttributeClockRate = 5,
  hipDeviceAttributeComputeMode = 6,
  hipDeviceAttributeComputePreemptionSupported = 7,
  hipDeviceAttributeConcurrentKernels = 8,
  hipDeviceAttributeConcurrentManagedAccess = 9,
  hipDeviceAttributeCooperativeLaunch = 10,
  hipDeviceAttributeCooperativeMultiDeviceLaunch = 11,
  hipDeviceAttributeDeviceOverlap = 12,
  hipDeviceAttributeDirectManagedMemAccessFromHost = 13,
  hipDeviceAttributeGlobalL1CacheSupported = 14,
  hipDeviceAttributeHostNativeAtomicSupported = 15,
  hipDeviceAttributeIntegrated = 16,
  hipDeviceAttributeIsMultiGpuBoard = 17,
  hipDeviceAttributeKernelExecTimeout = 18,
  hipDeviceAttributeL2CacheSize = 19,
  hipDeviceAttributeLocalL1CacheSupported = 20,
  hipDeviceAttributeLuid = 21,
  hipDeviceAttributeLuidDeviceNodeMask = 22,
  hipDeviceAttributeComputeCapabilityMajor = 23,
  hipDeviceAttributeManagedMemory = 24,
  hipDeviceAttributeMaxBlocksPerMultiProcessor = 25,
  hipDeviceAttributeMaxBlockDimX = 26,
  hipDeviceAttributeMaxBlockDimY = 27,
  hipDeviceAttributeMaxBlockDimZ = 28,
  hipDeviceAttributeMaxGridDimX = 29,
  hipDeviceAttributeMaxGridDimY = 30,
  hipDeviceAttributeMaxGridDimZ = 31,
  hipDeviceAttributeMaxSurface1D = 32,
  hipDeviceAttributeMaxSurface1DLayered = 33,
  hipDeviceAttributeMaxSurface2D = 34,
  hipDeviceAttributeMaxSurface2DLayered = 35,
  hipDeviceAttributeMaxSurface3D = 36,
  hipDeviceAttributeMaxSurfaceCubemap = 37,
  hipDeviceAttributeMaxSurfaceCubemapLayered = 38,
  hipDeviceAttributeMaxTexture1DWidth = 39,
  hipDeviceAttributeMaxTexture1DLayered = 40,
  hipDeviceAttributeMaxTexture1DLinear = 41,
  hipDeviceAttributeMaxTexture1DMipmap = 42,
  hipDeviceAttributeMaxTexture2DWidth = 43,
  hipDeviceAttributeMaxTexture2DHeight = 44,
  hipDeviceAttributeMaxTexture2DGather = 45,
  hipDeviceAttributeMaxTexture2DLayered = 46,
  hipDeviceAttributeMaxTexture2DLinear = 47,
  hipDeviceAttributeMaxTexture2DMipmap = 48,
  hipDeviceAttributeMaxTexture3DWidth = 49,
  hipDeviceAttributeMaxTexture3DHeight = 50,
  hipDeviceAttributeMaxTexture3DDepth = 51,
  hipDeviceAttributeMaxTexture3DAlt = 52,
  hipDeviceAttributeMaxTextureCubemap = 53,
  hipDeviceAttributeMaxTextureCubemapLayered = 54,
  hipDeviceAttributeMaxThreadsDim = 55,
  hipDeviceAttributeMaxThreadsPerBlock = 56,
  hipDeviceAttributeMaxThreadsPerMultiProcessor = 57,
  hipDeviceAttributeMaxPitch = 58,
  hipDeviceAttributeMemoryBusWidth = 59,
  hipDeviceAttributeMemoryClockRate = 60,
  hipDeviceAttributeComputeCapabilityMinor = 61,
  hipDeviceAttributeMultiGpuBoardGroupID = 62,
  hipDeviceAttributeMultiprocessorCount = 63,
  hipDeviceAttributeUnused1 = 64,
  hipDeviceAttributePageableMemoryAccess = 65,
  hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 66,
  hipDeviceAttributePciBusId = 67,
  hipDeviceAttributePciDeviceId = 68,
  hipDeviceAttributePciDomainID = 69,
  hipDeviceAttributePersistingL2CacheMaxSize = 70,
  hipDeviceAttributeMaxRegistersPerBlock = 71,
  hipDeviceAttributeMaxRegistersPerMultiprocessor = 72,
  hipDeviceAttributeReservedSharedMemPerBlock = 73,
  hipDeviceAttributeMaxSharedMemoryPerBlock = 74,
  hipDeviceAttributeSharedMemPerBlockOptin = 75,
  hipDeviceAttributeSharedMemPerMultiprocessor = 76,
  hipDeviceAttributeSingleToDoublePrecisionPerfRatio = 77,
  hipDeviceAttributeStreamPrioritiesSupported = 78,
  hipDeviceAttributeSurfaceAlignment = 79,
  hipDeviceAttributeTccDriver = 80,
  hipDeviceAttributeTextureAlignment = 81,
  hipDeviceAttributeTexturePitchAlignment = 82,
  hipDeviceAttributeTotalConstantMemory = 83,
  hipDeviceAttributeTotalGlobalMem = 84,
  hipDeviceAttributeUnifiedAddressing = 85,
  hipDeviceAttributeUnused2 = 86,
  hipDeviceAttributeWarpSize = 87,
  hipDeviceAttributeMemoryPoolsSupported = 88,
  hipDeviceAttributeVirtualMemoryManagementSupported = 89,
  hipDeviceAttributeHostRegisterSupported = 90,
  hipDeviceAttributeCudaCompatibleEnd = 9999,

  // AMD-specific attributes
  hipDeviceAttributeAmdSpecificBegin = 0x10000,
  hipDeviceAttributeClockInstructionRate = 0x10000,
  hipDeviceAttributeUnused3 = 0x10001,
  hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 0x10002,
  hipDeviceAttributeUnused4 = 0x10003,
  hipDeviceAttributeUnused5 = 0x10004,
  hipDeviceAttributeHdpMemFlushCntl = 0x10005,
  hipDeviceAttributeHdpRegFlushCntl = 0x10006,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = 0x10007,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = 0x10008,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = 0x10009,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = 0x1000a,
  hipDeviceAttributeIsLargeBar = 0x1000b,
  hipDeviceAttributeAsicRevision = 0x1000c,
  hipDeviceAttributeCanUseStreamWaitValue = 0x1000d,
  hipDeviceAttributeImageSupport = 0x1000e,
  hipDeviceAttributePhysicalMultiProcessorCount = 0x1000f,
  hipDeviceAttributeFineGrainSupport = 0x10010,
  hipDeviceAttributeWallClockRate = 0x10011,
  hipDeviceAttributeAmdSpecificEnd = 0x1ffff
} hipDeviceAttribute_t;

typedef enum hipMemoryType {
  hipMemoryTypeHost = 1,
  hipMemoryTypeDevice = 2,
  hipMemoryTypeArray = 3,
  hipMemoryTypeUnified = 4
} hipMemoryType;

typedef enum hipStreamFlags {
  hipStreamDefault = 0x00,
  hipStreamNonBlocking = 0x01
} hipStreamFlags_t;

// Host register flags.
typedef enum hipHostRegisterFlags {
  hipHostRegisterDefault = 0x00,
  hipHostRegisterPortable = 0x01,
  hipHostRegisterMapped = 0x02,
  hipHostRegisterIoMemory = 0x04,
  hipHostRegisterReadOnly = 0x08
} hipHostRegisterFlags_t;

typedef enum hipEventFlags {
  hipEventDefault = 0x00,
  hipEventBlockingSync = 0x01,
  hipEventDisableTiming = 0x02,
  hipEventInterprocess = 0x04,
  hipEventReleaseToDevice = 0x40000000,
  hipEventReleaseToSystem = 0x80000000
} hipEventFlags_t;

typedef enum hipDeviceP2PAttr {
  hipDevP2PAttrPerformanceRank = 0,
  hipDevP2PAttrAccessSupported = 1,
  hipDevP2PAttrNativeAtomicSupported = 2,
  hipDevP2PAttrHipArrayAccessSupported = 3
} hipDeviceP2PAttr;

typedef enum hipFuncAttribute {
  hipFuncAttributeMaxThreadsPerBlock = 0,
  hipFuncAttributeSharedSizeBytes = 1,
  hipFuncAttributeConstSizeBytes = 2,
  hipFuncAttributeLocalSizeBytes = 3,
  hipFuncAttributeNumRegs = 4,
  hipFuncAttributePtxVersion = 5,
  hipFuncAttributeBinaryVersion = 6,
  hipFuncAttributeCacheModeCA = 7,
  hipFuncAttributeMaxDynamicSharedSizeBytes = 8,
  hipFuncAttributePreferredSharedMemoryCarveout = 9,
  hipFuncAttributeMax
} hipFuncAttribute_t;

typedef enum hipLimit_t {
  hipLimitStackSize = 0x00,
  hipLimitPrintfFifoSize = 0x01,
  hipLimitMallocHeapSize = 0x02,
  hipLimitDevRuntimeSyncDepth = 0x03,
  hipLimitDevRuntimePendingLaunchCount = 0x04,
  hipLimitMaxL2FetchGranularity = 0x05,
  hipLimitPersistingL2CacheSize = 0x06,
  hipLimitRange
} hipLimit_t;

typedef enum hipFuncCache {
  hipFuncCachePreferNone = 0,
  hipFuncCachePreferShared = 1,
  hipFuncCachePreferL1 = 2,
  hipFuncCachePreferEqual = 3
} hipFuncCache_t;

typedef enum hipSharedMemConfig {
  hipSharedMemBankSizeDefault = 0,
  hipSharedMemBankSizeFourByte = 1,
  hipSharedMemBankSizeEightByte = 2
} hipSharedMemConfig;

typedef struct hipFuncAttributes {
  int binaryVersion;
  int cacheModeCA;
  size_t constSizeBytes;
  size_t localSizeBytes;
  int maxDynamicSharedSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int preferredShmemCarveout;
  int ptxVersion;
  size_t sharedSizeBytes;
} hipFuncAttributes;

typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
} hipMemcpyKind;

typedef struct hipIpcEventHandle_st {
  char reserved[64];
} hipIpcEventHandle_t;

typedef struct hipIpcMemHandle_st {
  char reserved[64];
} hipIpcMemHandle_t;

typedef struct hipUUID_st {
  unsigned char bytes[16];
} hipUUID;

typedef struct {
  // 32-bit integer atomics for global memory.
  unsigned hasGlobalInt32Atomics : 1;
  // 32-bit float atomic exch for global memory.
  unsigned hasGlobalFloatAtomicExch : 1;
  // 32-bit integer atomics for shared memory.
  unsigned hasSharedInt32Atomics : 1;
  // 32-bit float atomic exch for shared memory.
  unsigned hasSharedFloatAtomicExch : 1;
  // 32-bit float atomic add in global and shared memory.
  unsigned hasFloatAtomicAdd : 1;

  // 64-bit integer atomics for global memory.
  unsigned hasGlobalInt64Atomics : 1;
  // 64-bit integer atomics for shared memory.
  unsigned hasSharedInt64Atomics : 1;

  // Double-precision floating point.
  unsigned hasDoubles : 1;

  // Warp vote instructions (__any, __all).
  unsigned hasWarpVote : 1;
  // Warp ballot instructions (__ballot).
  unsigned hasWarpBallot : 1;
  // Warp shuffle operations. (__shfl_*).
  unsigned hasWarpShuffle : 1;
  // Funnel two words into one with shift&mask caps.
  unsigned hasFunnelShift : 1;

  // __threadfence_system.
  unsigned hasThreadFenceSystem : 1;
  // __syncthreads_count, syncthreads_and, syncthreads_or.
  unsigned hasSyncThreadsExt : 1;

  // Surface functions.
  unsigned hasSurfaceFuncs : 1;
  // Grid and group dims are 3D (rather than 2D).
  unsigned has3dGrid : 1;
  // Dynamic parallelism.
  unsigned hasDynamicParallelism : 1;
} hipDeviceArch_t;

typedef struct hipDeviceProp_t {
  // Device name.
  char name[256];
  // UUID of a device.
  hipUUID uuid;
  // 8-byte unique identifier. Only valid on windows.
  char luid[8];
  // LUID node mask
  unsigned int luidDeviceNodeMask;
  // Size of global memory region (in bytes).
  size_t totalGlobalMem;
  // Size of shared memory region (in bytes).
  size_t sharedMemPerBlock;
  // Registers per block.
  int regsPerBlock;
  // Warp size.
  int warpSize;
  // Maximum pitch in bytes allowed by memory copies pitched memory.
  size_t memPitch;
  // Max work items per work group or workgroup max size.
  int maxThreadsPerBlock;
  // Max number of threads in each dimension (XYZ) of a block.
  int maxThreadsDim[3];
  // Max grid dimensions (XYZ).
  int maxGridSize[3];
  // Max clock frequency of the multiProcessors in khz.
  int clockRate;
  // Size of shared memory region (in bytes).
  size_t totalConstMem;
  // Major compute capability. On HCC, this is an approximation and features
  // may differ from CUDA CC. See the arch feature flags for portable ways to
  // query feature caps.
  int major;
  // Minor compute capability. On HCC, this is an approximation and features
  // may differ from CUDA CC. See the arch feature flags for portable ways to
  // query feature caps.
  int minor;
  // Alignment requirement for textures.
  size_t textureAlignment;
  // Pitch alignment requirement for texture references bound to.
  size_t texturePitchAlignment;
  // Deprecated. Use asyncEngineCount instead.
  int deviceOverlap;
  // Number of multi-processors (compute units).
  int multiProcessorCount;
  // Run time limit for kernels executed on the device.
  int kernelExecTimeoutEnabled;
  // APU vs dGPU.
  int integrated;
  // Check whether HIP can map host memory.
  int canMapHostMemory;
  // Compute mode.
  int computeMode;
  // Maximum number of elements in 1D images.
  int maxTexture1D;
  // Maximum 1D mipmap texture size.
  int maxTexture1DMipmap;
  // Maximum size for 1D textures bound to linear memory.
  int maxTexture1DLinear;
  // Maximum dimensions (width, height) of 2D images, in image elements.
  int maxTexture2D[2];
  // Maximum number of elements in 2D array mipmap of images.
  int maxTexture2DMipmap[2];
  // Maximum 2D tex dimensions if tex are bound to pitched memory.
  int maxTexture2DLinear[3];
  // Maximum 2D tex dimensions if gather has to be performed.
  int maxTexture2DGather[2];
  // Maximum dimensions (width, height, depth) of 3D images, in image
  // elements.
  int maxTexture3D[3];
  // Maximum alternate 3D texture dims.
  int maxTexture3DAlt[3];
  // Maximum cubemap texture dims.
  int maxTextureCubemap;
  // Maximum number of elements in 1D array images.
  int maxTexture1DLayered[2];
  // Maximum number of elements in 2D array images.
  int maxTexture2DLayered[3];
  // Maximum cubemaps layered texture dims.
  int maxTextureCubemapLayered[2];
  // Maximum 1D surface size.
  int maxSurface1D;
  // Maximum 2D surface size.
  int maxSurface2D[2];
  // Maximum 3D surface size.
  int maxSurface3D[3];
  // Maximum 1D layered surface size.
  int maxSurface1DLayered[2];
  // Maximum 2D layared surface size.
  int maxSurface2DLayered[3];
  // Maximum cubemap surface size.
  int maxSurfaceCubemap;
  // Maximum cubemap layered surface size.
  int maxSurfaceCubemapLayered[2];
  // Alignment requirement for surface.
  size_t surfaceAlignment;
  // Device can possibly execute multiple kernels concurrently.
  int concurrentKernels;
  // Device has ECC support enabled.
  int ECCEnabled;
  // PCI Bus ID.
  int pciBusID;
  // PCI Device ID.
  int pciDeviceID;
  // PCI Domain ID.
  int pciDomainID;
  // 1:If device is Tesla device using TCC driver, else 0.
  int tccDriver;
  // Number of async engines.
  int asyncEngineCount;
  // Does device and host share unified address space.
  int unifiedAddressing;
  // Max global memory clock frequency in khz.
  int memoryClockRate;
  // Global memory bus width in bits.
  int memoryBusWidth;
  // L2 cache size.
  int l2CacheSize;
  // Device's max L2 persisting lines in bytes.
  int persistingL2CacheMaxSize;
  // Maximum resident threads per multi-processor.
  int maxThreadsPerMultiProcessor;
  // Device supports stream priority.
  int streamPrioritiesSupported;
  // Indicates globals are cached in L1.
  int globalL1CacheSupported;
  // Locals are cached in L1.
  int localL1CacheSupported;
  // Amount of shared memory available per multiprocessor.
  size_t sharedMemPerMultiprocessor;
  // registers available per multiprocessor.
  int regsPerMultiprocessor;
  // Device supports allocating managed memory on this system.
  int managedMemory;
  // 1 if device is on a multi-GPU board, 0 if not.
  int isMultiGpuBoard;
  // Unique identifier for a group of devices on same multiboard GPU.
  int multiGpuBoardGroupID;
  // Link between host and device supports native atomics.
  int hostNativeAtomicSupported;
  // Deprecated. CUDA only.
  int singleToDoublePrecisionPerfRatio;
  // Device supports coherently accessing pageable memory without calling
  // hipHostRegister on it.
  int pageableMemoryAccess;
  // Device can coherently access managed memory concurrently with the CPU.
  int concurrentManagedAccess;
  // Is compute preemption supported on the device.
  int computePreemptionSupported;
  // Device can access host registered memory with same address as the host.
  int canUseHostPointerForRegisteredMem;
  // HIP device supports cooperative launch.
  int cooperativeLaunch;
  // HIP device supports cooperative launch on multiple devices.
  int cooperativeMultiDeviceLaunch;
  // Per device m ax shared mem per block usable by special opt in.
  size_t sharedMemPerBlockOptin;
  // Device accesses pageable memory via the host's page tables.
  int pageableMemoryAccessUsesHostPageTables;
  // Host can directly access managed memory on the device without migration.
  int directManagedMemAccessFromHost;
  // Max number of blocks on CU.
  int maxBlocksPerMultiProcessor;
  // Max value of access policy window.
  int accessPolicyMaxWindowSize;
  // Shared memory reserved by driver per block.
  size_t reservedSharedMemPerBlock;
  // Device supports hipHostRegister.
  int hostRegisterSupported;
  // Indicates if device supports sparse hip arrays.
  int sparseHipArraySupported;
  // Device supports using the hipHostRegisterReadOnly flag with
  // hipHostRegister.
  int hostRegisterReadOnlySupported;
  // Indicates external timeline semaphore support.
  int timelineSemaphoreInteropSupported;
  // Indicates if device supports hipMallocAsync and hipMemPool APIs.
  int memoryPoolsSupported;
  // Indicates device support of RDMA APIs.
  int gpuDirectRDMASupported;
  // Bitmask to be interpreted according to
  // hipFlushGPUDirectRDMAWritesOptions.
  unsigned int gpuDirectRDMAFlushWritesOptions;
  // value of hipGPUDirectRDMAWritesOrdering.
  int gpuDirectRDMAWritesOrdering;
  // Bitmask of handle types support with mempool based IPC
  unsigned int memoryPoolSupportedHandleTypes;
  // Device supports deferred mapping HIP arrays and HIP mipmapped arrays.
  int deferredMappingHipArraySupported;
  // Device supports IPC events.
  int ipcEventSupported;
  // Device supports cluster launch.
  int clusterLaunch;
  // Indicates device supports unified function pointers.
  int unifiedFunctionPointers;
  // CUDA Reserved.
  int reserved[63];

  // Reserved for adding new entries for HIP/CUDA.
  int hipReserved[32];

  /* HIP Only struct members */

  // AMD GCN Arch Name. HIP Only.
  char gcnArchName[256];
  // Maximum Shared Memory Per CU. HIP Only.
  size_t maxSharedMemoryPerMultiProcessor;
  // Frequency in khz of the timer used by the device-side "clock*"
  // instructions. New for HIP.
  int clockInstructionRate;
  // Architectural feature flags.  New for HIP.
  hipDeviceArch_t arch;
  // Address of HDP_MEM_COHERENCY_FLUSH_CNTL register.
  unsigned int* hdpMemFlushCntl;
  // Address of HDP_REG_COHERENCY_FLUSH_CNTL register.
  unsigned int* hdpRegFlushCntl;
  // HIP device supports cooperative launch on multiple devices with unmatched
  // functions.
  int cooperativeMultiDeviceUnmatchedFunc;
  // HIP device supports cooperative launch on multiple devices with unmatched
  // grid dimensions.
  int cooperativeMultiDeviceUnmatchedGridDim;
  // HIP device supports cooperative launch on multiple devices with unmatched
  // block dimensions.
  int cooperativeMultiDeviceUnmatchedBlockDim;
  // HIP device supports cooperative launch on multiple devices with unmatched
  // shared memories.
  int cooperativeMultiDeviceUnmatchedSharedMem;
  // 1: if it is a large PCI bar device, else 0.
  int isLargeBar;
  // Revision of the GPU in this device.
  int asicRevision;
} hipDeviceProp_t;

typedef enum hipJitOption {
  hipJitOptionMaxRegisters = 0,
  hipJitOptionThreadsPerBlock = 1,
  hipJitOptionWallTime = 2,
  hipJitOptionInfoLogBuffer = 3,
  hipJitOptionInfoLogBufferSizeBytes = 4,
  hipJitOptionErrorLogBuffer = 5,
  hipJitOptionErrorLogBufferSizeBytes = 6,
  hipJitOptionOptimizationLevel = 7,
  hipJitOptionTargetFromContext = 8,
  hipJitOptionTarget = 9,
  hipJitOptionFallbackStrategy = 10,
  hipJitOptionGenerateDebugInfo = 11,
  hipJitOptionLogVerbose = 12,
  hipJitOptionGenerateLineInfo = 13,
  hipJitOptionCacheMode = 14,
  hipJitOptionSm3xOpt = 15,
  hipJitOptionFastCompile = 16,
  hipJitOptionNumOptions
} hipJitOption;

typedef void (*hipHostFn_t)(void* userData);

typedef size_t (*hipOccupancyB2DSize)(int blockSize);

// Graph types.
typedef struct hipGraph_st* hipGraph_t;
typedef struct hipGraphExec_st* hipGraphExec_t;
typedef struct hipGraphNode_st* hipGraphNode_t;

// Memory advice enum.
typedef enum hipMemAdvise_enum {
  hipMemAdviseSetReadMostly = 1,
  hipMemAdviseUnsetReadMostly = 2,
  hipMemAdviseSetPreferredLocation = 3,
  hipMemAdviseUnsetPreferredLocation = 4,
  hipMemAdviseSetAccessedBy = 5,
  hipMemAdviseUnsetAccessedBy = 6,
} hipMemAdvise_t;

// Pointer attribute enum.
typedef enum hipPointer_attribute {
  HIP_POINTER_ATTRIBUTE_CONTEXT = 1,
  HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
  HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
  HIP_POINTER_ATTRIBUTE_HOST_POINTER = 4,
  HIP_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
  HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
  HIP_POINTER_ATTRIBUTE_BUFFER_ID = 7,
  HIP_POINTER_ATTRIBUTE_IS_MANAGED = 8,
  HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
  HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10,
  HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
  HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
  HIP_POINTER_ATTRIBUTE_MAPPED = 13,
  HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
  HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
  HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
  HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
} hipPointer_attribute_t;

// Memory range attribute enum.
typedef enum hipMemRangeAttribute {
  hipMemRangeAttributeReadMostly = 1,
  hipMemRangeAttributePreferredLocation = 2,
  hipMemRangeAttributeAccessedBy = 3,
  hipMemRangeAttributeLastPrefetchLocation = 4,
} hipMemRangeAttribute;

// Stream capture mode.
typedef enum hipStreamCaptureMode {
  hipStreamCaptureModeGlobal = 0,
  hipStreamCaptureModeThreadLocal = 1,
  hipStreamCaptureModeRelaxed = 2,
} hipStreamCaptureMode;

// Stream capture status.
typedef enum hipStreamCaptureStatus {
  hipStreamCaptureStatusNone = 0,
  hipStreamCaptureStatusActive = 1,
  hipStreamCaptureStatusInvalidated = 2,
} hipStreamCaptureStatus;

// Graph node type.
typedef enum hipGraphNodeType {
  hipGraphNodeTypeKernel = 0,
  hipGraphNodeTypeMemcpy = 1,
  hipGraphNodeTypeMemset = 2,
  hipGraphNodeTypeHost = 3,
  hipGraphNodeTypeGraph = 4,
  hipGraphNodeTypeEmpty = 5,
  hipGraphNodeTypeWaitEvent = 6,
  hipGraphNodeTypeEventRecord = 7,
  hipGraphNodeTypeExtSemasSignal = 8,
  hipGraphNodeTypeExtSemasWait = 9,
  hipGraphNodeTypeMemAlloc = 10,
  hipGraphNodeTypeMemFree = 11,
} hipGraphNodeType;

// Graph instantiate flags.
typedef enum hipGraphInstantiate_flags {
  hipGraphInstantiateFlagAutoFreeOnLaunch = 1,
  hipGraphInstantiateFlagUpload = 2,
  hipGraphInstantiateFlagDeviceLaunch = 4,
  hipGraphInstantiateFlagUseNodePriority = 8,
} hipGraphInstantiate_flags;

// Stream capture dependency update flags.
#define hipStreamAddCaptureDependencies 0x1
#define hipStreamSetCaptureDependencies 0x2

//===----------------------------------------------------------------------===//
// Graph node parameter structures
//===----------------------------------------------------------------------===//

// Kernel launch parameter markers (same as CUDA).
#define HIP_LAUNCH_PARAM_END ((void*)0x00)
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)

// Kernel node parameters.
typedef struct hipKernelNodeParams {
  void* func;                   // Kernel function pointer.
  dim3 gridDim;                 // Grid dimensions.
  dim3 blockDim;                // Block dimensions.
  unsigned int sharedMemBytes;  // Dynamic shared memory size.
  void** kernelParams;          // Array of kernel parameters.
  void** extra;                 // Extra options.
} hipKernelNodeParams;

// Memory copy node parameters.
typedef struct hipMemcpy3DParms {
  hipArray_t srcArray;  // Source array.
  struct {
    size_t x, y, z;
  } srcPos;              // Source position.
  hipPitchedPtr srcPtr;  // Source pitched pointer.

  hipArray_t dstArray;  // Destination array.
  struct {
    size_t x, y, z;
  } dstPos;              // Destination position.
  hipPitchedPtr dstPtr;  // Destination pitched pointer.

  struct {
    size_t width, height, depth;
  } extent;            // Copy extent.
  hipMemcpyKind kind;  // Copy kind.
} hipMemcpy3DParms;

// Memset node parameters.
typedef struct hipMemsetParams {
  void* dst;                 // Destination pointer.
  unsigned int value;        // Value to set.
  unsigned int elementSize;  // Element size (1, 2, or 4 bytes).
  size_t width;              // Width in elements.
  size_t height;             // Height in elements.
  size_t pitch;              // Pitch in bytes.
} hipMemsetParams;

// Host node parameters.
typedef struct hipHostNodeParams {
  hipHostFn_t fn;  // Host function.
  void* userData;  // User data.
} hipHostNodeParams;

//===----------------------------------------------------------------------===//
// HIP API function declarations (exported from hip_hal.c)
//===----------------------------------------------------------------------===//

// Initialization
HIPAPI hipError_t hipInit(unsigned int flags);
HIPAPI hipError_t hipHALDeinit(void);  // HAL extension
HIPAPI hipError_t hipDriverGetVersion(int* driverVersion);
HIPAPI hipError_t hipRuntimeGetVersion(int* runtimeVersion);

// Device management
HIPAPI hipError_t hipGetDevice(int* device);
HIPAPI hipError_t hipSetDevice(int device);
HIPAPI hipError_t hipGetDeviceCount(int* count);
HIPAPI hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
HIPAPI hipError_t hipDeviceGetName(char* name, int len, hipDevice_t dev);
HIPAPI hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t dev);
HIPAPI hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t dev);
HIPAPI hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attrib,
                                        hipDevice_t dev);
HIPAPI hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device);
HIPAPI hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, hipDevice_t dev,
                                         hipDevice_t peerDev);
HIPAPI hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attrib,
                                           int srcDevice, int dstDevice);
HIPAPI hipError_t hipDeviceSynchronize(void);
HIPAPI hipError_t hipDeviceReset(void);

// Primary context
HIPAPI hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
HIPAPI hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);
HIPAPI hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev,
                                              unsigned int flags);
HIPAPI hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev,
                                              unsigned int* flags, int* active);
HIPAPI hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);

// Context management
HIPAPI hipError_t hipCtxCreate(hipCtx_t* pctx, unsigned int flags,
                               hipDevice_t dev);
HIPAPI hipError_t hipCtxDestroy(hipCtx_t ctx);
HIPAPI hipError_t hipCtxPushCurrent(hipCtx_t ctx);
HIPAPI hipError_t hipCtxPopCurrent(hipCtx_t* pctx);
HIPAPI hipError_t hipCtxSetCurrent(hipCtx_t ctx);
HIPAPI hipError_t hipCtxGetCurrent(hipCtx_t* pctx);
HIPAPI hipError_t hipCtxGetDevice(hipDevice_t* device);
HIPAPI hipError_t hipCtxSynchronize(void);
HIPAPI hipError_t hipCtxEnablePeerAccess(hipCtx_t peerContext,
                                         unsigned int flags);
HIPAPI hipError_t hipCtxDisablePeerAccess(hipCtx_t peerContext);
HIPAPI hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit);
HIPAPI hipError_t hipDeviceSetLimit(hipLimit_t limit, size_t value);

// Module management
HIPAPI hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
HIPAPI hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
HIPAPI hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                      unsigned int numOptions,
                                      hipJitOption* options,
                                      void** optionValues);
HIPAPI hipError_t hipModuleUnload(hipModule_t hmod);
HIPAPI hipError_t hipModuleGetFunction(hipFunction_t* hfunc, hipModule_t hmod,
                                       const char* name);
HIPAPI hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
                                     hipModule_t hmod, const char* name);

// Memory management
HIPAPI hipError_t hipMemGetInfo(size_t* free, size_t* total);
HIPAPI hipError_t hipMalloc(hipDeviceptr_t* dptr, size_t bytesize);
HIPAPI hipError_t hipMallocPitch(void** devPtr, size_t* pitch, size_t width,
                                 size_t height);
HIPAPI hipError_t hipFree(hipDeviceptr_t dptr);
HIPAPI hipError_t hipMallocHost(void** pp, size_t bytesize);
HIPAPI hipError_t hipFreeHost(void* p);
HIPAPI hipError_t hipHostAlloc(void** pp, size_t bytesize, unsigned int flags);
HIPAPI hipError_t hipHostGetDevicePointer(hipDeviceptr_t* pdptr, void* p,
                                          unsigned int flags);
HIPAPI hipError_t hipMallocManaged(hipDeviceptr_t* dptr, size_t bytesize,
                                   unsigned int flags);
HIPAPI hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags);
HIPAPI hipError_t hipHostUnregister(void* ptr);
HIPAPI hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                        hipDeviceptr_t dptr);
HIPAPI hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
HIPAPI hipError_t hipMemPtrGetInfo(void* ptr, size_t* size);

// Memory transfers
HIPAPI hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                            hipMemcpyKind kind);
HIPAPI hipError_t hipMemcpyWithStream(void* dst, const void* src,
                                      size_t sizeBytes, hipMemcpyKind kind,
                                      hipStream_t stream);
HIPAPI hipError_t hipMemcpyPeer(hipDeviceptr_t dstDevice, hipCtx_t dstContext,
                                hipDeviceptr_t srcDevice, hipCtx_t srcContext,
                                size_t ByteCount);
HIPAPI hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src,
                                size_t sizeBytes);
HIPAPI hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src,
                                size_t sizeBytes);
HIPAPI hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src,
                                size_t sizeBytes);

// Async memory transfers
HIPAPI hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                                 hipMemcpyKind kind, hipStream_t stream);
HIPAPI hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src,
                                     size_t sizeBytes, hipStream_t stream);
HIPAPI hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src,
                                     size_t sizeBytes, hipStream_t stream);
HIPAPI hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src,
                                     size_t sizeBytes, hipStream_t stream);

// Memory set
HIPAPI hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
HIPAPI hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes,
                                 hipStream_t stream);
HIPAPI hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value,
                              size_t count);
HIPAPI hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value,
                               size_t count);
HIPAPI hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count);
HIPAPI hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value,
                                   size_t count, hipStream_t stream);
HIPAPI hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value,
                                    size_t count, hipStream_t stream);
HIPAPI hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                                    hipStream_t stream);

// Stream management
HIPAPI hipError_t hipStreamCreate(hipStream_t* phStream);
HIPAPI hipError_t hipStreamCreateWithFlags(hipStream_t* phStream,
                                           unsigned int flags);
HIPAPI hipError_t hipStreamCreateWithPriority(hipStream_t* phStream,
                                              unsigned int flags, int priority);
HIPAPI hipError_t hipStreamWaitEvent(hipStream_t hStream, hipEvent_t hEvent,
                                     unsigned int flags);
HIPAPI hipError_t hipStreamQuery(hipStream_t hStream);
HIPAPI hipError_t hipStreamSynchronize(hipStream_t hStream);
HIPAPI hipError_t hipStreamDestroy(hipStream_t hStream);
HIPAPI hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);
HIPAPI hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);
HIPAPI hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device);

// Event management
HIPAPI hipError_t hipEventCreate(hipEvent_t* phEvent);
HIPAPI hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
HIPAPI hipError_t hipEventRecord(hipEvent_t hEvent, hipStream_t hStream);
HIPAPI hipError_t hipEventQuery(hipEvent_t hEvent);
HIPAPI hipError_t hipEventSynchronize(hipEvent_t hEvent);
HIPAPI hipError_t hipEventDestroy(hipEvent_t hEvent);
HIPAPI hipError_t hipEventElapsedTime(float* pMilliseconds, hipEvent_t hStart,
                                      hipEvent_t hEnd);

// Function management
HIPAPI hipError_t hipFuncGetAttribute(int* pi, hipFuncAttribute_t attrib,
                                      hipFunction_t hfunc);
HIPAPI hipError_t hipFuncGetAttributes(hipFuncAttributes* attr,
                                       hipFunction_t hfunc);
HIPAPI hipError_t hipFuncSetAttribute(hipFunction_t hfunc,
                                      hipFuncAttribute_t attrib, int value);
HIPAPI hipError_t hipFuncSetCacheConfig(hipFunction_t hfunc,
                                        hipFuncCache_t config);
HIPAPI hipError_t hipFuncSetSharedMemConfig(hipFunction_t hfunc,
                                            hipSharedMemConfig config);

// Execution control
HIPAPI hipError_t hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t hStream,
    void** kernelParams, void** extra);
HIPAPI hipError_t hipModuleLaunchCooperativeKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t hStream,
    void** kernelParams);
HIPAPI hipError_t hipLaunchHostFunc(hipStream_t hStream, hipHostFn_t fn,
                                    void* userData);

// Occupancy functions
HIPAPI hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk);
HIPAPI hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk,
    unsigned int flags);
HIPAPI hipError_t hipModuleOccupancyMaxPotentialBlockSize(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk,
    int blockSizeLimit);
HIPAPI hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk,
    int blockSizeLimit, unsigned int flags);

// Unified memory management
HIPAPI hipError_t hipMemAdvise(const void* dev_ptr, size_t count,
                               hipMemAdvise_t advice, int device);
HIPAPI hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count,
                                      int device, hipStream_t stream);
HIPAPI hipError_t hipPointerGetAttribute(void* data,
                                         hipPointer_attribute_t attribute,
                                         hipDeviceptr_t ptr);
HIPAPI hipError_t hipPointerSetAttribute(const void* value,
                                         hipPointer_attribute_t attribute,
                                         hipDeviceptr_t ptr);
HIPAPI hipError_t hipPointerGetAttributes(unsigned int numAttributes,
                                          hipPointer_attribute_t* attributes,
                                          void** data, const void* ptr);
HIPAPI hipError_t hipMemRangeGetAttribute(void* data, size_t data_size,
                                          hipMemRangeAttribute attribute,
                                          const void* dev_ptr, size_t count);
HIPAPI hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                           hipMemRangeAttribute* attributes,
                                           size_t num_attributes,
                                           const void* dev_ptr, size_t count);

// HIP graphs
HIPAPI hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags);
HIPAPI hipError_t hipGraphDestroy(hipGraph_t graph);
HIPAPI hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec,
                                      hipGraph_t graph,
                                      hipGraphNode_t* pErrorNode,
                                      char* pLogBuffer, size_t bufferSize);
HIPAPI hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec,
                                               hipGraph_t graph,
                                               unsigned long long flags);
HIPAPI hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);
HIPAPI hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);
HIPAPI hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec,
                                     hipGraph_t hGraph,
                                     hipGraphNode_t* hErrorNode_out,
                                     unsigned int flags);
HIPAPI hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pNodeParams);
HIPAPI hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pCopyParams);
HIPAPI hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pMemsetParams);
HIPAPI hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode,
                                      hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies,
                                      size_t numDependencies,
                                      const void* pNodeParams);
HIPAPI hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode,
                                       hipGraph_t graph,
                                       const hipGraphNode_t* pDependencies,
                                       size_t numDependencies);

// Stream capture
HIPAPI hipError_t hipStreamBeginCapture(hipStream_t stream,
                                        hipStreamCaptureMode mode);
HIPAPI hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph);
HIPAPI hipError_t hipStreamIsCapturing(hipStream_t stream,
                                       hipStreamCaptureStatus* pCaptureStatus);
HIPAPI hipError_t hipStreamGetCaptureInfo(
    hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId);
HIPAPI hipError_t hipStreamUpdateCaptureDependencies(
    hipStream_t stream, hipGraphNode_t* dependencies, size_t numDependencies,
    unsigned int flags);

//===----------------------------------------------------------------------===//
// Memory pool types and definitions
//===----------------------------------------------------------------------===//

// Memory pool handle type.
typedef struct hipMemPool_st* hipMemPool_t;

// Memory allocation handle types.
typedef enum hipMemAllocationHandleType {
  hipMemHandleTypeNone = 0,
  hipMemHandleTypePosixFileDescriptor = 1,
  hipMemHandleTypeWin32 = 2,
  hipMemHandleTypeWin32Kmt = 4,
} hipMemAllocationHandleType;

// Memory allocation types.
typedef enum hipMemAllocationType {
  hipMemAllocationTypeInvalid = 0,
  hipMemAllocationTypePinned = 1,
  hipMemAllocationTypeMax = 0x7FFFFFFF
} hipMemAllocationType;

// Memory location types.
typedef enum hipMemLocationType {
  hipMemLocationTypeInvalid = 0,
  hipMemLocationTypeDevice = 1,
  hipMemLocationTypeHost = 2,
  hipMemLocationTypeHostNuma = 3,
  hipMemLocationTypeHostNumaCurrent = 4,
} hipMemLocationType;

// Memory location descriptor.
typedef struct hipMemLocation {
  hipMemLocationType type;
  int id;
} hipMemLocation;

// Memory pool properties.
typedef struct hipMemPoolProps {
  hipMemAllocationType allocType;
  hipMemAllocationHandleType handleTypes;
  hipMemLocation location;
  void* win32SecurityAttributes;
  size_t maxSize;
  unsigned char reserved[16];
} hipMemPoolProps;

// Memory pool pointer export data.
typedef struct hipMemPoolPtrExportData {
  unsigned char reserved[64];
} hipMemPoolPtrExportData;

// Memory pool attributes.
typedef enum hipMemPool_attribute {
  hipMemPoolAttrReuseFollowEventDependencies = 1,
  hipMemPoolAttrReuseAllowOpportunistic = 2,
  hipMemPoolAttrReuseAllowInternalDependencies = 3,
  hipMemPoolAttrReleaseThreshold = 4,
  hipMemPoolAttrReservedMemCurrent = 5,
  hipMemPoolAttrReservedMemHigh = 6,
  hipMemPoolAttrUsedMemCurrent = 7,
  hipMemPoolAttrUsedMemHigh = 8,
} hipMemPool_attribute;

// Memory access flags.
typedef enum hipMemAccessFlags {
  hipMemAccessFlagsProtNone = 0,
  hipMemAccessFlagsProtRead = 1,
  hipMemAccessFlagsProtReadWrite = 3,
} hipMemAccessFlags;

// Memory access descriptor.
typedef struct hipMemAccessDesc {
  hipMemLocation location;
  hipMemAccessFlags flags;
} hipMemAccessDesc;

//===----------------------------------------------------------------------===//
// Memory pool API function declarations
//===----------------------------------------------------------------------===//

// Memory pool management
HIPAPI hipError_t hipMemPoolCreate(hipMemPool_t* pool,
                                   const hipMemPoolProps* poolProps);
HIPAPI hipError_t hipMemPoolDestroy(hipMemPool_t pool);
HIPAPI hipError_t hipMemPoolSetAttribute(hipMemPool_t pool,
                                         hipMemPool_attribute attr,
                                         void* value);
HIPAPI hipError_t hipMemPoolGetAttribute(hipMemPool_t pool,
                                         hipMemPool_attribute attr,
                                         void* value);
HIPAPI hipError_t hipMemPoolSetAccess(hipMemPool_t pool,
                                      const hipMemAccessDesc* map,
                                      size_t count);
HIPAPI hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags,
                                      hipMemPool_t pool,
                                      hipMemLocation* location);
HIPAPI hipError_t hipMemPoolTrimTo(hipMemPool_t pool, size_t minBytesToKeep);
HIPAPI hipError_t hipMemPoolExportToShareableHandle(
    void* handle_out, hipMemPool_t pool, hipMemAllocationHandleType handleType,
    unsigned int flags);
HIPAPI hipError_t hipMemPoolImportFromShareableHandle(
    hipMemPool_t* pool_out, void* handle, hipMemAllocationHandleType handleType,
    unsigned int flags);
HIPAPI hipError_t
hipMemPoolExportPointer(hipMemPoolPtrExportData* shareData_out, void* ptr);
HIPAPI hipError_t hipMemPoolImportPointer(void** ptr_out, hipMemPool_t pool,
                                          hipMemPoolPtrExportData* shareData);

// Device memory pool management
HIPAPI hipError_t hipDeviceSetMemPool(int device, hipMemPool_t pool);
HIPAPI hipError_t hipDeviceGetMemPool(hipMemPool_t* pool, int device);
HIPAPI hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* pool_out,
                                             int device);

// Async memory allocation
HIPAPI hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
HIPAPI hipError_t hipMallocFromPoolAsync(void** ptr, size_t size,
                                         hipMemPool_t pool, hipStream_t stream);
HIPAPI hipError_t hipFreeAsync(void* ptr, hipStream_t stream);

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

HIPAPI const char* hipGetErrorString(hipError_t error);
HIPAPI const char* hipGetErrorName(hipError_t error);
HIPAPI hipError_t hipGetLastError(void);
HIPAPI hipError_t hipPeekAtLastError(void);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_BINDING_HIP_API_H_
