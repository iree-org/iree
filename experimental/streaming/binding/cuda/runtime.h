// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_BINDING_CUDA_RUNTIME_H_
#define IREE_EXPERIMENTAL_STREAMING_BINDING_CUDA_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#include "driver.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// CUDA Runtime API types and definitions
//===----------------------------------------------------------------------===//

// CUDA Runtime API Version - CUDA 13.0
#define CUDART_VERSION 13000

// Forward declarations
typedef struct cudaStream_st* cudaStream_t;
typedef struct cudaEvent_st* cudaEvent_t;
typedef struct cudaArray* cudaArray_t;
typedef struct cudaArray* cudaArray_const_t;
typedef struct cudaMipmappedArray* cudaMipmappedArray_t;
typedef struct cudaMipmappedArray* cudaMipmappedArray_const_t;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;
typedef struct cudaExternalMemory_st* cudaExternalMemory_t;
typedef struct cudaExternalSemaphore_st* cudaExternalSemaphore_t;

// dim3 type for kernel launch dimensions.
typedef struct dim3 {
  unsigned int x, y, z;
#ifdef __cplusplus
  dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
#endif
} dim3;

// CUDA error types
typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorCudartUnloading = 4,
  cudaErrorProfilerDisabled = 5,
  cudaErrorProfilerNotInitialized = 6,
  cudaErrorProfilerAlreadyStarted = 7,
  cudaErrorProfilerAlreadyStopped = 8,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidPitchValue = 12,
  cudaErrorInvalidSymbol = 13,
  cudaErrorInvalidHostPointer = 16,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidTexture = 18,
  cudaErrorInvalidTextureBinding = 19,
  cudaErrorInvalidChannelDescriptor = 20,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorAddressOfConstant = 22,
  cudaErrorTextureFetchFailed = 23,
  cudaErrorTextureNotBound = 24,
  cudaErrorSynchronizationError = 25,
  cudaErrorInvalidFilterSetting = 26,
  cudaErrorInvalidNormSetting = 27,
  cudaErrorMixedDeviceExecution = 28,
  cudaErrorNotYetImplemented = 31,
  cudaErrorMemoryValueTooLarge = 32,
  cudaErrorStubLibrary = 34,
  cudaErrorInsufficientDriver = 35,
  cudaErrorCallRequiresNewerDriver = 36,
  cudaErrorInvalidSurface = 37,
  cudaErrorDuplicateVariableName = 43,
  cudaErrorDuplicateTextureName = 44,
  cudaErrorDuplicateSurfaceName = 45,
  cudaErrorDevicesUnavailable = 46,
  cudaErrorIncompatibleDriverContext = 49,
  cudaErrorMissingConfiguration = 52,
  cudaErrorPriorLaunchFailure = 53,
  cudaErrorLaunchMaxDepthExceeded = 65,
  cudaErrorLaunchFileScopedTex = 66,
  cudaErrorLaunchFileScopedSurf = 67,
  cudaErrorSyncDepthExceeded = 68,
  cudaErrorLaunchPendingCountExceeded = 69,
  cudaErrorInvalidDeviceFunction = 98,
  cudaErrorNoDevice = 100,
  cudaErrorInvalidDevice = 101,
  cudaErrorDeviceNotLicensed = 102,
  cudaErrorSoftwareValidityNotEstablished = 103,
  cudaErrorStartupFailure = 127,
  cudaErrorInvalidKernelImage = 200,
  cudaErrorDeviceUninitialized = 201,
  cudaErrorMapBufferObjectFailed = 205,
  cudaErrorUnmapBufferObjectFailed = 206,
  cudaErrorArrayIsMapped = 207,
  cudaErrorAlreadyMapped = 208,
  cudaErrorNoKernelImageForDevice = 209,
  cudaErrorAlreadyAcquired = 210,
  cudaErrorNotMapped = 211,
  cudaErrorNotMappedAsArray = 212,
  cudaErrorNotMappedAsPointer = 213,
  cudaErrorECCUncorrectable = 214,
  cudaErrorUnsupportedLimit = 215,
  cudaErrorDeviceAlreadyInUse = 216,
  cudaErrorPeerAccessUnsupported = 217,
  cudaErrorInvalidPtx = 218,
  cudaErrorInvalidGraphicsContext = 219,
  cudaErrorNvlinkUncorrectable = 220,
  cudaErrorJitCompilerNotFound = 221,
  cudaErrorUnsupportedPtxVersion = 222,
  cudaErrorJitCompilationDisabled = 223,
  cudaErrorUnsupportedExecAffinity = 224,
  cudaErrorUnsupportedDevSideSync = 225,
  cudaErrorContained = 226,
  cudaErrorInvalidSource = 300,
  cudaErrorFileNotFound = 301,
  cudaErrorSharedObjectSymbolNotFound = 302,
  cudaErrorSharedObjectInitFailed = 303,
  cudaErrorOperatingSystem = 304,
  cudaErrorInvalidResourceHandle = 400,
  cudaErrorIllegalState = 401,
  cudaErrorLossyQuery = 402,
  cudaErrorSymbolNotFound = 500,
  cudaErrorNotReady = 600,
  cudaErrorIllegalAddress = 700,
  cudaErrorLaunchOutOfResources = 701,
  cudaErrorLaunchTimeout = 702,
  cudaErrorLaunchIncompatibleTexturing = 703,
  cudaErrorPeerAccessAlreadyEnabled = 704,
  cudaErrorPeerAccessNotEnabled = 705,
  cudaErrorSetOnActiveProcess = 708,
  cudaErrorContextIsDestroyed = 709,
  cudaErrorAssert = 710,
  cudaErrorTooManyPeers = 711,
  cudaErrorHostMemoryAlreadyRegistered = 712,
  cudaErrorHostMemoryNotRegistered = 713,
  cudaErrorHardwareStackError = 714,
  cudaErrorIllegalInstruction = 715,
  cudaErrorMisalignedAddress = 716,
  cudaErrorInvalidAddressSpace = 717,
  cudaErrorInvalidPc = 718,
  cudaErrorLaunchFailure = 719,
  cudaErrorCooperativeLaunchTooLarge = 720,
  cudaErrorTensorMemoryLeak = 721,
  cudaErrorNotPermitted = 800,
  cudaErrorNotSupported = 801,
  cudaErrorSystemNotReady = 802,
  cudaErrorSystemDriverMismatch = 803,
  cudaErrorCompatNotSupportedOnDevice = 804,
  cudaErrorMpsConnectionFailed = 805,
  cudaErrorMpsRpcFailure = 806,
  cudaErrorMpsServerNotReady = 807,
  cudaErrorMpsMaxClientsReached = 808,
  cudaErrorMpsMaxConnectionsReached = 809,
  cudaErrorMpsClientTerminated = 810,
  cudaErrorCdpNotSupported = 811,
  cudaErrorCdpVersionMismatch = 812,
  cudaErrorStreamCaptureUnsupported = 900,
  cudaErrorStreamCaptureInvalidated = 901,
  cudaErrorStreamCaptureMerge = 902,
  cudaErrorStreamCaptureUnmatched = 903,
  cudaErrorStreamCaptureUnjoined = 904,
  cudaErrorStreamCaptureIsolation = 905,
  cudaErrorStreamCaptureImplicit = 906,
  cudaErrorCapturedEvent = 907,
  cudaErrorStreamCaptureWrongThread = 908,
  cudaErrorTimeout = 909,
  cudaErrorGraphExecUpdateFailure = 910,
  cudaErrorExternalDevice = 911,
  cudaErrorInvalidClusterSize = 912,
  cudaErrorFunctionNotLoaded = 913,
  cudaErrorInvalidResourceType = 914,
  cudaErrorInvalidResourceConfiguration = 915,
  cudaErrorUnknown = 999
} cudaError_t;

// Memory copy directions
typedef enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
} cudaMemcpyKind;

// Device flags
#define cudaDeviceScheduleAuto 0x00
#define cudaDeviceScheduleSpin 0x01
#define cudaDeviceScheduleYield 0x02
#define cudaDeviceScheduleBlockingSync 0x04
#define cudaDeviceScheduleMask 0x07
#define cudaDeviceMapHost 0x08
#define cudaDeviceLmemResizeToMax 0x10
#define cudaDeviceSyncMemops 0x80
#define cudaDeviceMask 0xff

// Stream flags
#define cudaStreamDefault 0x00
#define cudaStreamNonBlocking 0x01

// Event flags
#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04
#define cudaEventRecordDefault 0x00
#define cudaEventRecordExternal 0x01
#define cudaEventWaitDefault 0x00
#define cudaEventWaitExternal 0x01

// Host allocation flags
#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostAllocWriteCombined 0x04

// Host register flags
#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterMapped 0x02
#define cudaHostRegisterIoMemory 0x04
#define cudaHostRegisterReadOnly 0x08

// Memory attach flags
#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost 0x02
#define cudaMemAttachSingle 0x04

// Peer access flags
#define cudaPeerAccessDefault 0x00

// CPU device ID
#define cudaCpuDeviceId ((int)-1)
#define cudaInvalidDeviceId ((int)-2)

// Limit types
typedef enum cudaLimit {
  cudaLimitStackSize = 0x00,
  cudaLimitPrintfFifoSize = 0x01,
  cudaLimitMallocHeapSize = 0x02,
  cudaLimitDevRuntimeSyncDepth = 0x03,
  cudaLimitDevRuntimePendingLaunchCount = 0x04,
  cudaLimitMaxL2FetchGranularity = 0x05,
  cudaLimitPersistingL2CacheSize = 0x06
} cudaLimit;

// Function cache configurations
typedef enum cudaFuncCache {
  cudaFuncCachePreferNone = 0,
  cudaFuncCachePreferShared = 1,
  cudaFuncCachePreferL1 = 2,
  cudaFuncCachePreferEqual = 3
} cudaFuncCache;

// Shared memory configurations
typedef enum cudaSharedMemConfig {
  cudaSharedMemBankSizeDefault = 0,
  cudaSharedMemBankSizeFourByte = 1,
  cudaSharedMemBankSizeEightByte = 2
} cudaSharedMemConfig;

// Driver entry point query result
typedef enum cudaDriverEntryPointQueryResult {
  cudaDriverEntryPointSuccess = 0,
  cudaDriverEntryPointSymbolNotFound = 1,
  cudaDriverEntryPointVersionNotSufficent = 2
} cudaDriverEntryPointQueryResult;

// Device attributes
typedef enum cudaDeviceAttr {
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxBlockDimX = 2,
  cudaDevAttrMaxBlockDimY = 3,
  cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxGridDimX = 5,
  cudaDevAttrMaxGridDimY = 6,
  cudaDevAttrMaxGridDimZ = 7,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrTotalConstantMemory = 9,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrMaxPitch = 11,
  cudaDevAttrMaxRegistersPerBlock = 12,
  cudaDevAttrClockRate = 13,
  cudaDevAttrTextureAlignment = 14,
  cudaDevAttrGpuOverlap = 15,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrKernelExecTimeout = 17,
  cudaDevAttrIntegrated = 18,
  cudaDevAttrCanMapHostMemory = 19,
  cudaDevAttrComputeMode = 20,
  cudaDevAttrMaxTexture1DWidth = 21,
  cudaDevAttrMaxTexture2DWidth = 22,
  cudaDevAttrMaxTexture2DHeight = 23,
  cudaDevAttrMaxTexture3DWidth = 24,
  cudaDevAttrMaxTexture3DHeight = 25,
  cudaDevAttrMaxTexture3DDepth = 26,
  cudaDevAttrMaxTexture2DLayeredWidth = 27,
  cudaDevAttrMaxTexture2DLayeredHeight = 28,
  cudaDevAttrMaxTexture2DLayeredLayers = 29,
  cudaDevAttrSurfaceAlignment = 30,
  cudaDevAttrConcurrentKernels = 31,
  cudaDevAttrEccEnabled = 32,
  cudaDevAttrPciBusId = 33,
  cudaDevAttrPciDeviceId = 34,
  cudaDevAttrTccDriver = 35,
  cudaDevAttrMemoryClockRate = 36,
  cudaDevAttrGlobalMemoryBusWidth = 37,
  cudaDevAttrL2CacheSize = 38,
  cudaDevAttrMaxThreadsPerMultiProcessor = 39,
  cudaDevAttrAsyncEngineCount = 40,
  cudaDevAttrUnifiedAddressing = 41,
  cudaDevAttrMaxTexture1DLayeredWidth = 42,
  cudaDevAttrMaxTexture1DLayeredLayers = 43,
  cudaDevAttrMaxTexture2DGatherWidth = 45,
  cudaDevAttrMaxTexture2DGatherHeight = 46,
  cudaDevAttrMaxTexture3DWidthAlt = 47,
  cudaDevAttrMaxTexture3DHeightAlt = 48,
  cudaDevAttrMaxTexture3DDepthAlt = 49,
  cudaDevAttrPciDomainId = 50,
  cudaDevAttrTexturePitchAlignment = 51,
  cudaDevAttrMaxTextureCubemapWidth = 52,
  cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
  cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
  cudaDevAttrMaxSurface1DWidth = 55,
  cudaDevAttrMaxSurface2DWidth = 56,
  cudaDevAttrMaxSurface2DHeight = 57,
  cudaDevAttrMaxSurface3DWidth = 58,
  cudaDevAttrMaxSurface3DHeight = 59,
  cudaDevAttrMaxSurface3DDepth = 60,
  cudaDevAttrMaxSurface1DLayeredWidth = 61,
  cudaDevAttrMaxSurface1DLayeredLayers = 62,
  cudaDevAttrMaxSurface2DLayeredWidth = 63,
  cudaDevAttrMaxSurface2DLayeredHeight = 64,
  cudaDevAttrMaxSurface2DLayeredLayers = 65,
  cudaDevAttrMaxSurfaceCubemapWidth = 66,
  cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
  cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
  cudaDevAttrMaxTexture1DLinearWidth = 69,
  cudaDevAttrMaxTexture2DLinearWidth = 70,
  cudaDevAttrMaxTexture2DLinearHeight = 71,
  cudaDevAttrMaxTexture2DLinearPitch = 72,
  cudaDevAttrMaxTexture2DMipmappedWidth = 73,
  cudaDevAttrMaxTexture2DMipmappedHeight = 74,
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
  cudaDevAttrMaxTexture1DMipmappedWidth = 77,
  cudaDevAttrStreamPrioritiesSupported = 78,
  cudaDevAttrGlobalL1CacheSupported = 79,
  cudaDevAttrLocalL1CacheSupported = 80,
  cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
  cudaDevAttrMaxRegistersPerMultiprocessor = 82,
  cudaDevAttrManagedMemory = 83,
  cudaDevAttrIsMultiGpuBoard = 84,
  cudaDevAttrMultiGpuBoardGroupID = 85,
  cudaDevAttrHostNativeAtomicSupported = 86,
  cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
  cudaDevAttrPageableMemoryAccess = 88,
  cudaDevAttrConcurrentManagedAccess = 89,
  cudaDevAttrComputePreemptionSupported = 90,
  cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
  cudaDevAttrReserved92 = 92,
  cudaDevAttrReserved93 = 93,
  cudaDevAttrReserved94 = 94,
  cudaDevAttrCooperativeLaunch = 95,
  cudaDevAttrReserved96 = 96,
  cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
  cudaDevAttrCanFlushRemoteWrites = 98,
  cudaDevAttrHostRegisterSupported = 99,
  cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
  cudaDevAttrDirectManagedMemAccessFromHost = 101,
  cudaDevAttrMaxBlocksPerMultiprocessor = 106,
  cudaDevAttrMaxPersistingL2CacheSize = 108,
  cudaDevAttrMaxAccessPolicyWindowSize = 109,
  cudaDevAttrReservedSharedMemoryPerBlock = 111,
  cudaDevAttrSparseCudaArraySupported = 112,
  cudaDevAttrHostRegisterReadOnlySupported = 113,
  cudaDevAttrTimelineSemaphoreInteropSupported = 114,
  cudaDevAttrMemoryPoolsSupported = 115,
  cudaDevAttrGPUDirectRDMASupported = 116,
  cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
  cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
  cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
  cudaDevAttrClusterLaunch = 120,
  cudaDevAttrDeferredMappingCudaArraySupported = 121,
  cudaDevAttrReserved122 = 122,
  cudaDevAttrReserved123 = 123,
  cudaDevAttrReserved124 = 124,
  cudaDevAttrIpcEventSupport = 125,
  cudaDevAttrMemSyncDomainCount = 126,
  cudaDevAttrReserved127 = 127,
  cudaDevAttrReserved128 = 128,
  cudaDevAttrReserved129 = 129,
  cudaDevAttrNumaConfig = 130,
  cudaDevAttrNumaId = 131,
  cudaDevAttrReserved132 = 132,
  cudaDevAttrMpsEnabled = 133,
  cudaDevAttrHostNumaId = 134,
  cudaDevAttrD3D12CigSupported = 135,
  cudaDevAttrVulkanCigSupported = 138,
  cudaDevAttrGpuPciDeviceId = 139,
  cudaDevAttrGpuPciSubsystemId = 140,
  cudaDevAttrReserved141 = 141,
  cudaDevAttrHostNumaMemoryPoolsSupported = 142,
  cudaDevAttrHostNumaMultinodeIpcSupported = 143,
  cudaDevAttrHostMemoryPoolsSupported = 144,
  cudaDevAttrReserved145 = 145,
  cudaDevAttrOnlyPartialHostNativeAtomicSupported = 147,
  cudaDevAttrMax
} cudaDeviceAttr;

// Memory advise values
typedef enum cudaMemoryAdvise {
  cudaMemAdviseSetReadMostly = 1,
  cudaMemAdviseUnsetReadMostly = 2,
  cudaMemAdviseSetPreferredLocation = 3,
  cudaMemAdviseUnsetPreferredLocation = 4,
  cudaMemAdviseSetAccessedBy = 5,
  cudaMemAdviseUnsetAccessedBy = 6
} cudaMemoryAdvise;

// Function attributes
typedef enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
  cudaFuncAttributePreferredSharedMemoryCarveout = 9,
  cudaFuncAttributeClusterDimMustBeSet = 10,
  cudaFuncAttributeRequiredClusterWidth = 11,
  cudaFuncAttributeRequiredClusterHeight = 12,
  cudaFuncAttributeRequiredClusterDepth = 13,
  cudaFuncAttributeNonPortableClusterSizeAllowed = 14,
  cudaFuncAttributeClusterSchedulingPolicyPreference = 15,
  cudaFuncAttributeMax
} cudaFuncAttribute;

// Device properties structure
typedef struct cudaDeviceProp {
  char name[256];
  // TODO: CUDA 13 UUID type
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  size_t sharedMemPerBlockOptin;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int maxBlocksPerMultiProcessor;
  int accessPolicyMaxWindowSize;
  size_t reservedSharedMemPerBlock;
  int hostRegisterSupported;
  int sparseHipArraySupported;
  int hostRegisterReadOnlySupported;
  int timelineSemaphoreInteropSupported;
  int memoryPoolsSupported;
  int gpuDirectRDMASupported;
  unsigned int gpuDirectRDMAFlushWritesOptions;
  int gpuDirectRDMAWritesOrdering;
  unsigned int memoryPoolSupportedHandleTypes;
  int deferredMappingHipArraySupported;
  int ipcEventSupported;
  int clusterLaunch;
  int unifiedFunctionPointers;
  int reserved[63];
  int cudaReserved[32];
} cudaDeviceProp;

// Function attributes
typedef struct cudaFuncAttributes {
  size_t sharedSizeBytes;
  size_t constSizeBytes;
  size_t localSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int ptxVersion;
  int binaryVersion;
  int cacheModeCA;
  int maxDynamicSharedSizeBytes;
  int preferredShmemCarveout;
  int clusterDimMustBeSet;
  int requiredClusterWidth;
  int requiredClusterHeight;
  int requiredClusterDepth;
  int clusterSchedulingPolicyPreference;
  int nonPortableClusterSizeAllowed;
  int reserved[16];
} cudaFuncAttributes;

// Pointer attributes
typedef struct cudaPointerAttributes {
  enum {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
  } type;
  int device;
  void* devicePointer;
  void* hostPointer;
  int isManaged;
} cudaPointerAttributes;

// Memory copy structures
typedef struct cudaPos {
  size_t x;
  size_t y;
  size_t z;
} cudaPos;

typedef struct cudaExtent {
  size_t width;
  size_t height;
  size_t depth;
} cudaExtent;

typedef struct cudaPitchedPtr {
  void* ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
} cudaPitchedPtr;

typedef struct cudaMemcpy3DParms {
  cudaArray_t srcArray;
  struct cudaPos srcPos;
  struct cudaPitchedPtr srcPtr;
  cudaArray_t dstArray;
  struct cudaPos dstPos;
  struct cudaPitchedPtr dstPtr;
  struct cudaExtent extent;
  cudaMemcpyKind kind;
} cudaMemcpy3DParms;

// Callback types
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status,
                                      void* userData);
typedef void (*cudaHostFn_t)(void* userData);

// IPC handles
typedef struct cudaIpcEventHandle_st {
  char reserved[64];
} cudaIpcEventHandle_t;

typedef struct cudaIpcMemHandle_st {
  char reserved[64];
} cudaIpcMemHandle_t;

// Memory pool types
typedef struct cudaMemPool_st* cudaMemPool_t;

typedef enum cudaMemPoolAttr {
  cudaMemPoolReuseFollowEventDependencies = 0x1,
  cudaMemPoolReuseAllowOpportunistic = 0x2,
  cudaMemPoolReuseAllowInternalDependencies = 0x3,
  cudaMemPoolAttrReleaseThreshold = 0x4,
  cudaMemPoolAttrReservedMemCurrent = 0x5,
  cudaMemPoolAttrReservedMemHigh = 0x6,
  cudaMemPoolAttrUsedMemCurrent = 0x7,
  cudaMemPoolAttrUsedMemHigh = 0x8
} cudaMemPoolAttr;

typedef struct cudaMemPoolProps {
  int allocType;
  int handleTypes;
  int location;
  void* win32SecurityAttributes;
  size_t maxSize;
  unsigned short usage;
  unsigned char reserved[54];
} cudaMemPoolProps;

typedef struct cudaMemPoolPtrExportData {
  unsigned char reserved[64];
} cudaMemPoolPtrExportData;

// Memory location types
typedef enum cudaMemLocationType {
  cudaMemLocationTypeInvalid = 0,
  cudaMemLocationTypeNone = 0,
  cudaMemLocationTypeDevice = 1,
  cudaMemLocationTypeHost = 2,
  cudaMemLocationTypeHostNuma = 3,
  cudaMemLocationTypeHostNumaCurrent = 4
} cudaMemLocationType;

typedef struct cudaMemLocation {
  cudaMemLocationType type;
  int id;
} cudaMemLocation;

typedef enum cudaMemAccessFlags {
  cudaMemAccessFlagsProtNone = 0,
  cudaMemAccessFlagsProtRead = 1,
  cudaMemAccessFlagsProtReadWrite = 3
} cudaMemAccessFlags;

typedef struct cudaMemAccessDesc {
  cudaMemLocation location;
  cudaMemAccessFlags flags;
} cudaMemAccessDesc;

// Occupancy calculator function
typedef size_t (*cudaOccupancyB2DSize)(int blockSize);

// Graph types
typedef struct cudaGraph_st* cudaGraph_t;
typedef struct cudaGraphExec_st* cudaGraphExec_t;
typedef struct cudaGraphNode_st* cudaGraphNode_t;

typedef enum cudaGraphNodeType {
  cudaGraphNodeTypeKernel = 0x00,
  cudaGraphNodeTypeMemcpy = 0x01,
  cudaGraphNodeTypeMemset = 0x02,
  cudaGraphNodeTypeHost = 0x03,
  cudaGraphNodeTypeGraph = 0x04,
  cudaGraphNodeTypeEmpty = 0x05,
  cudaGraphNodeTypeWaitEvent = 0x06,
  cudaGraphNodeTypeEventRecord = 0x07,
  cudaGraphNodeTypeExtSemaphoreSignal = 0x08,
  cudaGraphNodeTypeExtSemaphoreWait = 0x09,
  cudaGraphNodeTypeMemAlloc = 0x0a,
  cudaGraphNodeTypeMemFree = 0x0b,
  cudaGraphNodeTypeConditional = 0x0d
} cudaGraphNodeType;

// Stream capture modes
typedef enum cudaStreamCaptureMode {
  cudaStreamCaptureModeGlobal = 0,
  cudaStreamCaptureModeThreadLocal = 1,
  cudaStreamCaptureModeRelaxed = 2
} cudaStreamCaptureMode;

typedef enum cudaStreamCaptureStatus {
  cudaStreamCaptureStatusNone = 0,
  cudaStreamCaptureStatusActive = 1,
  cudaStreamCaptureStatusInvalidated = 2
} cudaStreamCaptureStatus;

typedef enum cudaGraphExecUpdateResult {
  cudaGraphExecUpdateSuccess = 0x0,
  cudaGraphExecUpdateError = 0x1,
  cudaGraphExecUpdateErrorTopologyChanged = 0x2,
  cudaGraphExecUpdateErrorNodeTypeChanged = 0x3,
  cudaGraphExecUpdateErrorFunctionChanged = 0x4,
  cudaGraphExecUpdateErrorParametersChanged = 0x5,
  cudaGraphExecUpdateErrorNotSupported = 0x6,
  cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x7,
  cudaGraphExecUpdateErrorAttributesChanged = 0x8
} cudaGraphExecUpdateResult;

typedef struct cudaGraphExecUpdateResultInfo {
  cudaGraphExecUpdateResult result;
  cudaGraphNode_t errorNode;
  cudaGraphNode_t errorFromNode;
} cudaGraphExecUpdateResultInfo;

typedef enum cudaGraphInstantiateResult {
  cudaGraphInstantiateSuccess = 0,
  cudaGraphInstantiateError = 1,
  cudaGraphInstantiateInvalidStructure = 2,
  cudaGraphInstantiateNodeOperationNotSupported = 3,
  cudaGraphInstantiateMultipleDevicesNotSupported = 4,
  cudaGraphInstantiateConditionalHandleUnused = 5
} cudaGraphInstantiateResult;

typedef struct cudaGraphInstantiateParams {
  unsigned long long flags;
  cudaStream_t uploadStream;
  cudaGraphNode_t errNode_out;
  cudaGraphInstantiateResult result_out;
} cudaGraphInstantiateParams;

// Graph edge data
typedef struct cudaGraphEdgeData {
  unsigned char from_port;
  unsigned char to_port;
  unsigned char type;
  unsigned char reserved[5];
} cudaGraphEdgeData;

// Graph node parameters
typedef struct cudaKernelNodeParams {
  void* func;
  dim3 gridDim;
  dim3 blockDim;
  unsigned int sharedMemBytes;
  void** kernelParams;
  void** extra;
} cudaKernelNodeParams;

typedef struct cudaMemsetParams {
  void* dst;
  size_t pitch;
  unsigned int value;
  unsigned int elementSize;
  size_t width;
  size_t height;
} cudaMemsetParams;

typedef struct cudaHostNodeParams {
  cudaHostFn_t fn;
  void* userData;
} cudaHostNodeParams;

//===----------------------------------------------------------------------===//
// Initialization and Version Management
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaDriverGetVersion(int* driverVersion);
CUDAAPI cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
CUDAAPI cudaError_t
cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, uint64_t flags,
                        cudaDriverEntryPointQueryResult* driverStatus);
CUDAAPI cudaError_t cudaGetDriverEntryPointByVersion(
    const char* symbol, void** funcPtr, unsigned int cudaVersion,
    uint64_t flags, cudaDriverEntryPointQueryResult* driverStatus);

//===----------------------------------------------------------------------===//
// Device Management
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaGetDevice(int* device);
CUDAAPI cudaError_t cudaSetDevice(int device);
CUDAAPI cudaError_t cudaGetDeviceCount(int* count);
CUDAAPI cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
CUDAAPI cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr,
                                           int device);
CUDAAPI cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool,
                                                int device);
CUDAAPI cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);
CUDAAPI cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device);
CUDAAPI cudaError_t cudaSetDeviceFlags(unsigned int flags);
CUDAAPI cudaError_t cudaGetDeviceFlags(unsigned int* flags);
CUDAAPI cudaError_t cudaDeviceReset(void);
CUDAAPI cudaError_t cudaDeviceSynchronize(void);
CUDAAPI cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
CUDAAPI cudaError_t cudaDeviceGetLimit(size_t* value, cudaLimit limit);
CUDAAPI cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device);
CUDAAPI cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId);
CUDAAPI cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* cacheConfig);
CUDAAPI cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);
CUDAAPI cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* config);
CUDAAPI cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);
CUDAAPI cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority,
                                                     int* greatestPriority);

//===----------------------------------------------------------------------===//
// Memory Management
//===----------------------------------------------------------------------===//

// Basic allocation
CUDAAPI cudaError_t cudaMalloc(void** devPtr, size_t size);
CUDAAPI cudaError_t cudaFree(void* devPtr);
CUDAAPI cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
CUDAAPI cudaError_t cudaMallocHost(void** ptr, size_t size);
CUDAAPI cudaError_t cudaFreeHost(void* ptr);
CUDAAPI cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
CUDAAPI cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width,
                                    size_t height);
CUDAAPI cudaError_t cudaMallocManaged(void** devPtr, size_t size,
                                      unsigned int flags);

// Memory transfer
CUDAAPI cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                               cudaMemcpyKind kind);
CUDAAPI cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                    cudaMemcpyKind kind, cudaStream_t stream);
CUDAAPI cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src,
                                 size_t spitch, size_t width, size_t height,
                                 cudaMemcpyKind kind);
CUDAAPI cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                                      size_t spitch, size_t width,
                                      size_t height, cudaMemcpyKind kind,
                                      cudaStream_t stream);
CUDAAPI cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src,
                                   int srcDevice, size_t count);
CUDAAPI cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice,
                                        const void* src, int srcDevice,
                                        size_t count, cudaStream_t stream);
CUDAAPI cudaError_t cudaMemset(void* devPtr, int value, size_t count);
CUDAAPI cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count,
                                    cudaStream_t stream);
CUDAAPI cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value,
                                 size_t width, size_t height);
CUDAAPI cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value,
                                      size_t width, size_t height,
                                      cudaStream_t stream);

// Pinned memory
CUDAAPI cudaError_t cudaHostRegister(void* ptr, size_t size,
                                     unsigned int flags);
CUDAAPI cudaError_t cudaHostUnregister(void* ptr);
CUDAAPI cudaError_t cudaHostGetDevicePointer(void** devicePtr, void* hostPtr,
                                             unsigned int flags);
CUDAAPI cudaError_t cudaHostGetFlags(unsigned int* flags, void* hostPtr);

// Unified memory
CUDAAPI cudaError_t cudaMemAdvise(const void* devPtr, size_t count,
                                  cudaMemoryAdvise advice, int device);
CUDAAPI cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count,
                                         cudaMemLocation location,
                                         unsigned int flags, cudaStream_t stream);

// Pointer queries
CUDAAPI cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes,
                                             const void* ptr);

//===----------------------------------------------------------------------===//
// Stream Management
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaStreamCreate(cudaStream_t* stream);
CUDAAPI cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream,
                                              unsigned int flags);
CUDAAPI cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream,
                                                 unsigned int flags,
                                                 int priority);
CUDAAPI cudaError_t cudaStreamDestroy(cudaStream_t stream);
CUDAAPI cudaError_t cudaStreamSynchronize(cudaStream_t stream);
CUDAAPI cudaError_t cudaStreamQuery(cudaStream_t stream);
CUDAAPI cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                        unsigned int flags);
CUDAAPI cudaError_t cudaStreamGetFlags(cudaStream_t stream,
                                       unsigned int* flags);
CUDAAPI cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority);
CUDAAPI cudaError_t cudaStreamGetId(cudaStream_t stream,
                                    unsigned long long* streamId);
CUDAAPI cudaError_t cudaStreamCopyAttributes(cudaStream_t dst,
                                             cudaStream_t src);
CUDAAPI cudaError_t cudaStreamGetCaptureInfo(
    cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out, cudaGraph_t* graph_out,
    const cudaGraphNode_t** dependencies_out,
    const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out);
CUDAAPI cudaError_t cudaStreamIsCapturing(
    cudaStream_t stream, cudaStreamCaptureStatus* captureStatus);
CUDAAPI cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                           cudaStreamCaptureMode mode);
CUDAAPI cudaError_t cudaStreamEndCapture(cudaStream_t stream,
                                         cudaGraph_t* graph);
CUDAAPI cudaError_t cudaStreamUpdateCaptureDependencies(
    cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies,
    unsigned int flags);
CUDAAPI cudaError_t
cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode);

//===----------------------------------------------------------------------===//
// Event Management
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaEventCreate(cudaEvent_t* event);
CUDAAPI cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event,
                                             unsigned int flags);
CUDAAPI cudaError_t cudaEventDestroy(cudaEvent_t event);
CUDAAPI cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
CUDAAPI cudaError_t cudaEventRecordWithFlags(cudaEvent_t event,
                                             cudaStream_t stream,
                                             unsigned int flags);
CUDAAPI cudaError_t cudaEventQuery(cudaEvent_t event);
CUDAAPI cudaError_t cudaEventSynchronize(cudaEvent_t event);
CUDAAPI cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start,
                                         cudaEvent_t end);

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaGetLastError(void);
CUDAAPI cudaError_t cudaPeekAtLastError(void);
CUDAAPI const char* cudaGetErrorString(cudaError_t error);
CUDAAPI const char* cudaGetErrorName(cudaError_t error);

//===----------------------------------------------------------------------===//
// Execution Control
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim,
                                     dim3 blockDim, void** args,
                                     size_t sharedMem, cudaStream_t stream);
CUDAAPI cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim,
                                                dim3 blockDim, void** args,
                                                size_t sharedMem,
                                                cudaStream_t stream);
CUDAAPI cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn,
                                       void* userData);
CUDAAPI cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr,
                                          const void* func);
CUDAAPI cudaError_t cudaFuncSetAttribute(const void* func,
                                         cudaFuncAttribute attr,
                                         int value);
CUDAAPI cudaError_t cudaFuncSetCacheConfig(const void* func,
                                           cudaFuncCache config);
CUDAAPI cudaError_t cudaFuncSetSharedMemConfig(const void* func,
                                               cudaSharedMemConfig config);
CUDAAPI cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSmemSize);
CUDAAPI cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSmemSize,
    unsigned int flags);
CUDAAPI cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int* minGridSize, int* blockSize, const void* func,
    cudaOccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSmemSize,
    int blockSizeLimit);
CUDAAPI cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(
    int* minGridSize, int* blockSize, const void* func,
    cudaOccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSmemSize,
    int blockSizeLimit, unsigned int flags);

//===----------------------------------------------------------------------===//
// Peer Access
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device,
                                            int peerDevice);
CUDAAPI cudaError_t cudaDeviceEnablePeerAccess(int peerDevice,
                                               unsigned int flags);
CUDAAPI cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);

//===----------------------------------------------------------------------===//
// Memory Pools
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool,
                                      const cudaMemPoolProps* poolProps);
CUDAAPI cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool);
CUDAAPI cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                            cudaMemPoolAttr attr, void* value);
CUDAAPI cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t pool,
                                            cudaMemPoolAttr attr, void* value);
CUDAAPI cudaError_t cudaMemPoolSetAccess(cudaMemPool_t pool,
                                         const cudaMemAccessDesc* map,
                                         size_t count);
CUDAAPI cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags* flags,
                                         cudaMemPool_t memPool,
                                         cudaMemLocation* location);
CUDAAPI cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool,
                                      size_t minBytesToKeep);
CUDAAPI cudaError_t cudaMallocAsync(void** ptr, size_t size,
                                    cudaStream_t stream);
CUDAAPI cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream);
CUDAAPI cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size,
                                            cudaMemPool_t pool,
                                            cudaStream_t stream);

//===----------------------------------------------------------------------===//
// Graph Management
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int flags);
CUDAAPI cudaError_t cudaGraphDestroy(cudaGraph_t graph);
CUDAAPI cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                         cudaGraph_t graph,
                                         unsigned long long flags);
CUDAAPI cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec,
                                                  cudaGraph_t graph,
                                                  unsigned long long flags);
CUDAAPI cudaError_t cudaGraphInstantiateWithParams(
    cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
    cudaGraphInstantiateParams* instantiateParams);
CUDAAPI cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
CUDAAPI cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec,
                                    cudaStream_t stream);
CUDAAPI cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec,
                                        cudaGraph_t hGraph,
                                        cudaGraphExecUpdateResultInfo* resultInfo);
CUDAAPI cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes,
                                      size_t* numNodes);

//===----------------------------------------------------------------------===//
// IPC (Inter-Process Communication)
//===----------------------------------------------------------------------===//

CUDAAPI cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle,
                                          cudaEvent_t event);
CUDAAPI cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event,
                                           cudaIpcEventHandle_t handle);
CUDAAPI cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle,
                                        void* devPtr);
CUDAAPI cudaError_t cudaIpcOpenMemHandle(void** devPtr,
                                         cudaIpcMemHandle_t handle,
                                         unsigned int flags);
CUDAAPI cudaError_t cudaIpcCloseMemHandle(void* devPtr);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_BINDING_CUDA_RUNTIME_H_
