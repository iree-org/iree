// automatically generated sources
#ifndef _HIP_PROF_STR_H
#define _HIP_PROF_STR_H
#define HIP_PROF_VER 1

// Dummy API primitives
#define INIT_NONE_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetAddress_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetBorderColor_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyDtoA_CB_ARGS_DATA(cb_data) {};
#define INIT_hipArrayGetDescriptor_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexObjectGetResourceViewDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyAtoHAsync_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDestroyTextureObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipArray3DGetDescriptor_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddress_CB_ARGS_DATA(cb_data) {};
#define INIT_hipArrayDestroy_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetMaxAnisotropy_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetMipmapFilterMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDeviceGetCount_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyArrayToArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTexture2D_CB_ARGS_DATA(cb_data) {};
#define INIT_hipCreateTextureObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyHtoAAsync_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyAtoA_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyAtoD_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTextureToMipmappedArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetMipmapLevelClamp_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTextureToArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFlags_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFormat_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexObjectGetTextureDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexObjectDestroy_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpy2DArrayToArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureReference_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMipmappedArrayDestroy_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetFilterMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetFormat_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyToArrayAsync_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddress2D_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectResourceViewDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetFlags_CB_ARGS_DATA(cb_data) {};
#define INIT_hipUnbindTexture_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetMipmapLevelBias_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFilterMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureAlignmentOffset_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMipmappedArrayGetLevel_CB_ARGS_DATA(cb_data) {};
#define INIT_hipCreateSurfaceObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMipmappedArrayCreate_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexObjectGetResourceDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetChannelDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetAddressMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectResourceDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipModuleLaunchKernelExt_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpy2DToArrayAsync_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetBorderColor_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDestroySurfaceObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetMipmapFilterMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetMaxAnisotropy_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexObjectCreate_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddressMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetMipmapLevelBias_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyFromArrayAsync_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTexture_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetMipmappedArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetMipmappedArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipSetValidDevices_CB_ARGS_DATA(cb_data) {};
#define INIT_ihipModuleLaunchKernel_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetMipmapLevelClamp_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectTextureDesc_CB_ARGS_DATA(cb_data) {};

// HIP API callbacks ID enumaration
enum hip_api_id_t {
  HIP_API_ID_hipDrvMemcpy3DAsync = 0,
  HIP_API_ID_hipDeviceEnablePeerAccess = 1,
  HIP_API_ID_hipFuncSetSharedMemConfig = 2,
  HIP_API_ID_hipMemcpyToSymbolAsync = 3,
  HIP_API_ID_hipMallocPitch = 4,
  HIP_API_ID_hipMalloc = 5,
  HIP_API_ID_hipMemsetD16 = 6,
  HIP_API_ID_hipExtStreamGetCUMask = 7,
  HIP_API_ID_hipEventRecord = 8,
  HIP_API_ID_hipCtxSynchronize = 9,
  HIP_API_ID_hipSetDevice = 10,
  HIP_API_ID_hipCtxGetApiVersion = 11,
  HIP_API_ID_hipMemcpyFromSymbolAsync = 12,
  HIP_API_ID_hipExtGetLinkTypeAndHopCount = 13,
  HIP_API_ID___hipPopCallConfiguration = 14,
  HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor = 15,
  HIP_API_ID_hipMemset3D = 16,
  HIP_API_ID_hipStreamCreateWithPriority = 17,
  HIP_API_ID_hipMemcpy2DToArray = 18,
  HIP_API_ID_hipMemsetD8Async = 19,
  HIP_API_ID_hipCtxGetCacheConfig = 20,
  HIP_API_ID_hipModuleGetFunction = 21,
  HIP_API_ID_hipStreamWaitEvent = 22,
  HIP_API_ID_hipDeviceGetStreamPriorityRange = 23,
  HIP_API_ID_hipModuleLoad = 24,
  HIP_API_ID_hipDevicePrimaryCtxSetFlags = 25,
  HIP_API_ID_hipLaunchCooperativeKernel = 26,
  HIP_API_ID_hipLaunchCooperativeKernelMultiDevice = 27,
  HIP_API_ID_hipMemcpyAsync = 28,
  HIP_API_ID_hipMalloc3DArray = 29,
  HIP_API_ID_hipMallocHost = 30,
  HIP_API_ID_hipCtxGetCurrent = 31,
  HIP_API_ID_hipDevicePrimaryCtxGetState = 32,
  HIP_API_ID_hipEventQuery = 33,
  HIP_API_ID_hipEventCreate = 34,
  HIP_API_ID_hipMemGetAddressRange = 35,
  HIP_API_ID_hipMemcpyFromSymbol = 36,
  HIP_API_ID_hipArrayCreate = 37,
  HIP_API_ID_hipStreamAttachMemAsync = 38,
  HIP_API_ID_hipStreamGetFlags = 39,
  HIP_API_ID_hipMallocArray = 40,
  HIP_API_ID_hipCtxGetSharedMemConfig = 41,
  HIP_API_ID_hipDeviceDisablePeerAccess = 42,
  HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize = 43,
  HIP_API_ID_hipMemPtrGetInfo = 44,
  HIP_API_ID_hipFuncGetAttribute = 45,
  HIP_API_ID_hipCtxGetFlags = 46,
  HIP_API_ID_hipStreamDestroy = 47,
  HIP_API_ID___hipPushCallConfiguration = 48,
  HIP_API_ID_hipMemset3DAsync = 49,
  HIP_API_ID_hipDeviceGetPCIBusId = 50,
  HIP_API_ID_hipInit = 51,
  HIP_API_ID_hipMemcpyAtoH = 52,
  HIP_API_ID_hipStreamGetPriority = 53,
  HIP_API_ID_hipMemset2D = 54,
  HIP_API_ID_hipMemset2DAsync = 55,
  HIP_API_ID_hipDeviceCanAccessPeer = 56,
  HIP_API_ID_hipLaunchByPtr = 57,
  HIP_API_ID_hipMemPrefetchAsync = 58,
  HIP_API_ID_hipCtxDestroy = 59,
  HIP_API_ID_hipMemsetD16Async = 60,
  HIP_API_ID_hipModuleUnload = 61,
  HIP_API_ID_hipHostUnregister = 62,
  HIP_API_ID_hipProfilerStop = 63,
  HIP_API_ID_hipExtStreamCreateWithCUMask = 64,
  HIP_API_ID_hipStreamSynchronize = 65,
  HIP_API_ID_hipFreeHost = 66,
  HIP_API_ID_hipDeviceSetCacheConfig = 67,
  HIP_API_ID_hipGetErrorName = 68,
  HIP_API_ID_hipMemcpyHtoD = 69,
  HIP_API_ID_hipModuleGetGlobal = 70,
  HIP_API_ID_hipMemcpyHtoA = 71,
  HIP_API_ID_hipCtxCreate = 72,
  HIP_API_ID_hipMemcpy2D = 73,
  HIP_API_ID_hipIpcCloseMemHandle = 74,
  HIP_API_ID_hipChooseDevice = 75,
  HIP_API_ID_hipDeviceSetSharedMemConfig = 76,
  HIP_API_ID_hipMallocMipmappedArray = 77,
  HIP_API_ID_hipSetupArgument = 78,
  HIP_API_ID_hipIpcGetEventHandle = 79,
  HIP_API_ID_hipFreeArray = 80,
  HIP_API_ID_hipCtxSetCacheConfig = 81,
  HIP_API_ID_hipFuncSetCacheConfig = 82,
  HIP_API_ID_hipLaunchKernel = 83,
  HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = 84,
  HIP_API_ID_hipModuleGetTexRef = 85,
  HIP_API_ID_hipFuncSetAttribute = 86,
  HIP_API_ID_hipEventElapsedTime = 87,
  HIP_API_ID_hipConfigureCall = 88,
  HIP_API_ID_hipMemAdvise = 89,
  HIP_API_ID_hipMemcpy3DAsync = 90,
  HIP_API_ID_hipEventDestroy = 91,
  HIP_API_ID_hipCtxPopCurrent = 92,
  HIP_API_ID_hipGetSymbolAddress = 93,
  HIP_API_ID_hipHostGetFlags = 94,
  HIP_API_ID_hipHostMalloc = 95,
  HIP_API_ID_hipCtxSetSharedMemConfig = 96,
  HIP_API_ID_hipFreeMipmappedArray = 97,
  HIP_API_ID_hipMemGetInfo = 98,
  HIP_API_ID_hipDeviceReset = 99,
  HIP_API_ID_hipMemset = 100,
  HIP_API_ID_hipMemsetD8 = 101,
  HIP_API_ID_hipMemcpyParam2DAsync = 102,
  HIP_API_ID_hipHostRegister = 103,
  HIP_API_ID_hipDriverGetVersion = 104,
  HIP_API_ID_hipArray3DCreate = 105,
  HIP_API_ID_hipIpcOpenMemHandle = 106,
  HIP_API_ID_hipGetLastError = 107,
  HIP_API_ID_hipGetDeviceFlags = 108,
  HIP_API_ID_hipDeviceGetSharedMemConfig = 109,
  HIP_API_ID_hipDrvMemcpy3D = 110,
  HIP_API_ID_hipMemcpy2DFromArray = 111,
  HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = 112,
  HIP_API_ID_hipSetDeviceFlags = 113,
  HIP_API_ID_hipHccModuleLaunchKernel = 114,
  HIP_API_ID_hipFree = 115,
  HIP_API_ID_hipOccupancyMaxPotentialBlockSize = 116,
  HIP_API_ID_hipDeviceGetAttribute = 117,
  HIP_API_ID_hipDeviceComputeCapability = 118,
  HIP_API_ID_hipCtxDisablePeerAccess = 119,
  HIP_API_ID_hipMallocManaged = 120,
  HIP_API_ID_hipDeviceGetByPCIBusId = 121,
  HIP_API_ID_hipIpcGetMemHandle = 122,
  HIP_API_ID_hipMemcpyHtoDAsync = 123,
  HIP_API_ID_hipCtxGetDevice = 124,
  HIP_API_ID_hipMemcpyDtoD = 125,
  HIP_API_ID_hipModuleLoadData = 126,
  HIP_API_ID_hipDevicePrimaryCtxRelease = 127,
  HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor = 128,
  HIP_API_ID_hipCtxSetCurrent = 129,
  HIP_API_ID_hipGetErrorString = 130,
  HIP_API_ID_hipStreamCreate = 131,
  HIP_API_ID_hipDevicePrimaryCtxRetain = 132,
  HIP_API_ID_hipDeviceGet = 133,
  HIP_API_ID_hipStreamCreateWithFlags = 134,
  HIP_API_ID_hipMemcpyFromArray = 135,
  HIP_API_ID_hipMemcpy2DAsync = 136,
  HIP_API_ID_hipFuncGetAttributes = 137,
  HIP_API_ID_hipGetSymbolSize = 138,
  HIP_API_ID_hipHostFree = 139,
  HIP_API_ID_hipEventCreateWithFlags = 140,
  HIP_API_ID_hipStreamQuery = 141,
  HIP_API_ID_hipMemcpy3D = 142,
  HIP_API_ID_hipMemcpyToSymbol = 143,
  HIP_API_ID_hipMemcpy = 144,
  HIP_API_ID_hipPeekAtLastError = 145,
  HIP_API_ID_hipExtLaunchMultiKernelMultiDevice = 146,
  HIP_API_ID_hipHostAlloc = 147,
  HIP_API_ID_hipStreamAddCallback = 148,
  HIP_API_ID_hipMemcpyToArray = 149,
  HIP_API_ID_hipMemsetD32 = 150,
  HIP_API_ID_hipExtModuleLaunchKernel = 151,
  HIP_API_ID_hipDeviceSynchronize = 152,
  HIP_API_ID_hipDeviceGetCacheConfig = 153,
  HIP_API_ID_hipMalloc3D = 154,
  HIP_API_ID_hipPointerGetAttributes = 155,
  HIP_API_ID_hipMemsetAsync = 156,
  HIP_API_ID_hipDeviceGetName = 157,
  HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags = 158,
  HIP_API_ID_hipCtxPushCurrent = 159,
  HIP_API_ID_hipMemcpyPeer = 160,
  HIP_API_ID_hipEventSynchronize = 161,
  HIP_API_ID_hipMemcpyDtoDAsync = 162,
  HIP_API_ID_hipProfilerStart = 163,
  HIP_API_ID_hipExtMallocWithFlags = 164,
  HIP_API_ID_hipCtxEnablePeerAccess = 165,
  HIP_API_ID_hipMemAllocHost = 166,
  HIP_API_ID_hipMemcpyDtoHAsync = 167,
  HIP_API_ID_hipModuleLaunchKernel = 168,
  HIP_API_ID_hipMemAllocPitch = 169,
  HIP_API_ID_hipExtLaunchKernel = 170,
  HIP_API_ID_hipMemcpy2DFromArrayAsync = 171,
  HIP_API_ID_hipDeviceGetLimit = 172,
  HIP_API_ID_hipModuleLoadDataEx = 173,
  HIP_API_ID_hipRuntimeGetVersion = 174,
  HIP_API_ID_hipMemRangeGetAttribute = 175,
  HIP_API_ID_hipDeviceGetP2PAttribute = 176,
  HIP_API_ID_hipMemcpyPeerAsync = 177,
  HIP_API_ID_hipGetDeviceProperties = 178,
  HIP_API_ID_hipMemcpyDtoH = 179,
  HIP_API_ID_hipMemcpyWithStream = 180,
  HIP_API_ID_hipDeviceTotalMem = 181,
  HIP_API_ID_hipHostGetDevicePointer = 182,
  HIP_API_ID_hipMemRangeGetAttributes = 183,
  HIP_API_ID_hipMemcpyParam2D = 184,
  HIP_API_ID_hipDevicePrimaryCtxReset = 185,
  HIP_API_ID_hipGetMipmappedArrayLevel = 186,
  HIP_API_ID_hipMemsetD32Async = 187,
  HIP_API_ID_hipGetDevice = 188,
  HIP_API_ID_hipGetDeviceCount = 189,
  HIP_API_ID_hipIpcOpenEventHandle = 190,
  HIP_API_ID_NUMBER = 191,

  HIP_API_ID_NONE = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetAddress = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetBorderColor = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyDtoA = HIP_API_ID_NUMBER,
  HIP_API_ID_hipArrayGetDescriptor = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexObjectGetResourceViewDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyAtoHAsync = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDestroyTextureObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipArray3DGetDescriptor = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddress = HIP_API_ID_NUMBER,
  HIP_API_ID_hipArrayDestroy = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetMaxAnisotropy = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetMipmapFilterMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDeviceGetCount = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyArrayToArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTexture2D = HIP_API_ID_NUMBER,
  HIP_API_ID_hipCreateTextureObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyHtoAAsync = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyAtoA = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyAtoD = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTextureToMipmappedArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetMipmapLevelClamp = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTextureToArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFlags = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFormat = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexObjectGetTextureDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexObjectDestroy = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpy2DArrayToArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureReference = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMipmappedArrayDestroy = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetFilterMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetFormat = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyToArrayAsync = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddress2D = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectResourceViewDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetFlags = HIP_API_ID_NUMBER,
  HIP_API_ID_hipUnbindTexture = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetMipmapLevelBias = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFilterMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureAlignmentOffset = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMipmappedArrayGetLevel = HIP_API_ID_NUMBER,
  HIP_API_ID_hipCreateSurfaceObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMipmappedArrayCreate = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexObjectGetResourceDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetChannelDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetAddressMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectResourceDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipModuleLaunchKernelExt = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpy2DToArrayAsync = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetBorderColor = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDestroySurfaceObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetMipmapFilterMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetMaxAnisotropy = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexObjectCreate = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddressMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetMipmapLevelBias = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyFromArrayAsync = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTexture = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetMipmappedArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetMipmappedArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipSetValidDevices = HIP_API_ID_NUMBER,
  HIP_API_ID_ihipModuleLaunchKernel = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetMipmapLevelClamp = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectTextureDesc = HIP_API_ID_NUMBER,
};

// Return HIP API string by given ID
static inline const char* hip_api_name(const uint32_t id) {
  switch(id) {
    case HIP_API_ID_hipDrvMemcpy3DAsync: return "hipDrvMemcpy3DAsync";
    case HIP_API_ID_hipDeviceEnablePeerAccess: return "hipDeviceEnablePeerAccess";
    case HIP_API_ID_hipFuncSetSharedMemConfig: return "hipFuncSetSharedMemConfig";
    case HIP_API_ID_hipMemcpyToSymbolAsync: return "hipMemcpyToSymbolAsync";
    case HIP_API_ID_hipMallocPitch: return "hipMallocPitch";
    case HIP_API_ID_hipMalloc: return "hipMalloc";
    case HIP_API_ID_hipMemsetD16: return "hipMemsetD16";
    case HIP_API_ID_hipExtStreamGetCUMask: return "hipExtStreamGetCUMask";
    case HIP_API_ID_hipEventRecord: return "hipEventRecord";
    case HIP_API_ID_hipCtxSynchronize: return "hipCtxSynchronize";
    case HIP_API_ID_hipSetDevice: return "hipSetDevice";
    case HIP_API_ID_hipCtxGetApiVersion: return "hipCtxGetApiVersion";
    case HIP_API_ID_hipMemcpyFromSymbolAsync: return "hipMemcpyFromSymbolAsync";
    case HIP_API_ID_hipExtGetLinkTypeAndHopCount: return "hipExtGetLinkTypeAndHopCount";
    case HIP_API_ID___hipPopCallConfiguration: return "__hipPopCallConfiguration";
    case HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor: return "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor";
    case HIP_API_ID_hipMemset3D: return "hipMemset3D";
    case HIP_API_ID_hipStreamCreateWithPriority: return "hipStreamCreateWithPriority";
    case HIP_API_ID_hipMemcpy2DToArray: return "hipMemcpy2DToArray";
    case HIP_API_ID_hipMemsetD8Async: return "hipMemsetD8Async";
    case HIP_API_ID_hipCtxGetCacheConfig: return "hipCtxGetCacheConfig";
    case HIP_API_ID_hipModuleGetFunction: return "hipModuleGetFunction";
    case HIP_API_ID_hipStreamWaitEvent: return "hipStreamWaitEvent";
    case HIP_API_ID_hipDeviceGetStreamPriorityRange: return "hipDeviceGetStreamPriorityRange";
    case HIP_API_ID_hipModuleLoad: return "hipModuleLoad";
    case HIP_API_ID_hipDevicePrimaryCtxSetFlags: return "hipDevicePrimaryCtxSetFlags";
    case HIP_API_ID_hipLaunchCooperativeKernel: return "hipLaunchCooperativeKernel";
    case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice: return "hipLaunchCooperativeKernelMultiDevice";
    case HIP_API_ID_hipMemcpyAsync: return "hipMemcpyAsync";
    case HIP_API_ID_hipMalloc3DArray: return "hipMalloc3DArray";
    case HIP_API_ID_hipMallocHost: return "hipMallocHost";
    case HIP_API_ID_hipCtxGetCurrent: return "hipCtxGetCurrent";
    case HIP_API_ID_hipDevicePrimaryCtxGetState: return "hipDevicePrimaryCtxGetState";
    case HIP_API_ID_hipEventQuery: return "hipEventQuery";
    case HIP_API_ID_hipEventCreate: return "hipEventCreate";
    case HIP_API_ID_hipMemGetAddressRange: return "hipMemGetAddressRange";
    case HIP_API_ID_hipMemcpyFromSymbol: return "hipMemcpyFromSymbol";
    case HIP_API_ID_hipArrayCreate: return "hipArrayCreate";
    case HIP_API_ID_hipStreamAttachMemAsync: return "hipStreamAttachMemAsync";
    case HIP_API_ID_hipStreamGetFlags: return "hipStreamGetFlags";
    case HIP_API_ID_hipMallocArray: return "hipMallocArray";
    case HIP_API_ID_hipCtxGetSharedMemConfig: return "hipCtxGetSharedMemConfig";
    case HIP_API_ID_hipDeviceDisablePeerAccess: return "hipDeviceDisablePeerAccess";
    case HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize: return "hipModuleOccupancyMaxPotentialBlockSize";
    case HIP_API_ID_hipMemPtrGetInfo: return "hipMemPtrGetInfo";
    case HIP_API_ID_hipFuncGetAttribute: return "hipFuncGetAttribute";
    case HIP_API_ID_hipCtxGetFlags: return "hipCtxGetFlags";
    case HIP_API_ID_hipStreamDestroy: return "hipStreamDestroy";
    case HIP_API_ID___hipPushCallConfiguration: return "__hipPushCallConfiguration";
    case HIP_API_ID_hipMemset3DAsync: return "hipMemset3DAsync";
    case HIP_API_ID_hipDeviceGetPCIBusId: return "hipDeviceGetPCIBusId";
    case HIP_API_ID_hipInit: return "hipInit";
    case HIP_API_ID_hipMemcpyAtoH: return "hipMemcpyAtoH";
    case HIP_API_ID_hipStreamGetPriority: return "hipStreamGetPriority";
    case HIP_API_ID_hipMemset2D: return "hipMemset2D";
    case HIP_API_ID_hipMemset2DAsync: return "hipMemset2DAsync";
    case HIP_API_ID_hipDeviceCanAccessPeer: return "hipDeviceCanAccessPeer";
    case HIP_API_ID_hipLaunchByPtr: return "hipLaunchByPtr";
    case HIP_API_ID_hipMemPrefetchAsync: return "hipMemPrefetchAsync";
    case HIP_API_ID_hipCtxDestroy: return "hipCtxDestroy";
    case HIP_API_ID_hipMemsetD16Async: return "hipMemsetD16Async";
    case HIP_API_ID_hipModuleUnload: return "hipModuleUnload";
    case HIP_API_ID_hipHostUnregister: return "hipHostUnregister";
    case HIP_API_ID_hipProfilerStop: return "hipProfilerStop";
    case HIP_API_ID_hipExtStreamCreateWithCUMask: return "hipExtStreamCreateWithCUMask";
    case HIP_API_ID_hipStreamSynchronize: return "hipStreamSynchronize";
    case HIP_API_ID_hipFreeHost: return "hipFreeHost";
    case HIP_API_ID_hipDeviceSetCacheConfig: return "hipDeviceSetCacheConfig";
    case HIP_API_ID_hipGetErrorName: return "hipGetErrorName";
    case HIP_API_ID_hipMemcpyHtoD: return "hipMemcpyHtoD";
    case HIP_API_ID_hipModuleGetGlobal: return "hipModuleGetGlobal";
    case HIP_API_ID_hipMemcpyHtoA: return "hipMemcpyHtoA";
    case HIP_API_ID_hipCtxCreate: return "hipCtxCreate";
    case HIP_API_ID_hipMemcpy2D: return "hipMemcpy2D";
    case HIP_API_ID_hipIpcCloseMemHandle: return "hipIpcCloseMemHandle";
    case HIP_API_ID_hipChooseDevice: return "hipChooseDevice";
    case HIP_API_ID_hipDeviceSetSharedMemConfig: return "hipDeviceSetSharedMemConfig";
    case HIP_API_ID_hipMallocMipmappedArray: return "hipMallocMipmappedArray";
    case HIP_API_ID_hipSetupArgument: return "hipSetupArgument";
    case HIP_API_ID_hipIpcGetEventHandle: return "hipIpcGetEventHandle";
    case HIP_API_ID_hipFreeArray: return "hipFreeArray";
    case HIP_API_ID_hipCtxSetCacheConfig: return "hipCtxSetCacheConfig";
    case HIP_API_ID_hipFuncSetCacheConfig: return "hipFuncSetCacheConfig";
    case HIP_API_ID_hipLaunchKernel: return "hipLaunchKernel";
    case HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: return "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags";
    case HIP_API_ID_hipModuleGetTexRef: return "hipModuleGetTexRef";
    case HIP_API_ID_hipFuncSetAttribute: return "hipFuncSetAttribute";
    case HIP_API_ID_hipEventElapsedTime: return "hipEventElapsedTime";
    case HIP_API_ID_hipConfigureCall: return "hipConfigureCall";
    case HIP_API_ID_hipMemAdvise: return "hipMemAdvise";
    case HIP_API_ID_hipMemcpy3DAsync: return "hipMemcpy3DAsync";
    case HIP_API_ID_hipEventDestroy: return "hipEventDestroy";
    case HIP_API_ID_hipCtxPopCurrent: return "hipCtxPopCurrent";
    case HIP_API_ID_hipGetSymbolAddress: return "hipGetSymbolAddress";
    case HIP_API_ID_hipHostGetFlags: return "hipHostGetFlags";
    case HIP_API_ID_hipHostMalloc: return "hipHostMalloc";
    case HIP_API_ID_hipCtxSetSharedMemConfig: return "hipCtxSetSharedMemConfig";
    case HIP_API_ID_hipFreeMipmappedArray: return "hipFreeMipmappedArray";
    case HIP_API_ID_hipMemGetInfo: return "hipMemGetInfo";
    case HIP_API_ID_hipDeviceReset: return "hipDeviceReset";
    case HIP_API_ID_hipMemset: return "hipMemset";
    case HIP_API_ID_hipMemsetD8: return "hipMemsetD8";
    case HIP_API_ID_hipMemcpyParam2DAsync: return "hipMemcpyParam2DAsync";
    case HIP_API_ID_hipHostRegister: return "hipHostRegister";
    case HIP_API_ID_hipDriverGetVersion: return "hipDriverGetVersion";
    case HIP_API_ID_hipArray3DCreate: return "hipArray3DCreate";
    case HIP_API_ID_hipIpcOpenMemHandle: return "hipIpcOpenMemHandle";
    case HIP_API_ID_hipGetLastError: return "hipGetLastError";
    case HIP_API_ID_hipGetDeviceFlags: return "hipGetDeviceFlags";
    case HIP_API_ID_hipDeviceGetSharedMemConfig: return "hipDeviceGetSharedMemConfig";
    case HIP_API_ID_hipDrvMemcpy3D: return "hipDrvMemcpy3D";
    case HIP_API_ID_hipMemcpy2DFromArray: return "hipMemcpy2DFromArray";
    case HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: return "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags";
    case HIP_API_ID_hipSetDeviceFlags: return "hipSetDeviceFlags";
    case HIP_API_ID_hipHccModuleLaunchKernel: return "hipHccModuleLaunchKernel";
    case HIP_API_ID_hipFree: return "hipFree";
    case HIP_API_ID_hipOccupancyMaxPotentialBlockSize: return "hipOccupancyMaxPotentialBlockSize";
    case HIP_API_ID_hipDeviceGetAttribute: return "hipDeviceGetAttribute";
    case HIP_API_ID_hipDeviceComputeCapability: return "hipDeviceComputeCapability";
    case HIP_API_ID_hipCtxDisablePeerAccess: return "hipCtxDisablePeerAccess";
    case HIP_API_ID_hipMallocManaged: return "hipMallocManaged";
    case HIP_API_ID_hipDeviceGetByPCIBusId: return "hipDeviceGetByPCIBusId";
    case HIP_API_ID_hipIpcGetMemHandle: return "hipIpcGetMemHandle";
    case HIP_API_ID_hipMemcpyHtoDAsync: return "hipMemcpyHtoDAsync";
    case HIP_API_ID_hipCtxGetDevice: return "hipCtxGetDevice";
    case HIP_API_ID_hipMemcpyDtoD: return "hipMemcpyDtoD";
    case HIP_API_ID_hipModuleLoadData: return "hipModuleLoadData";
    case HIP_API_ID_hipDevicePrimaryCtxRelease: return "hipDevicePrimaryCtxRelease";
    case HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor: return "hipOccupancyMaxActiveBlocksPerMultiprocessor";
    case HIP_API_ID_hipCtxSetCurrent: return "hipCtxSetCurrent";
    case HIP_API_ID_hipGetErrorString: return "hipGetErrorString";
    case HIP_API_ID_hipStreamCreate: return "hipStreamCreate";
    case HIP_API_ID_hipDevicePrimaryCtxRetain: return "hipDevicePrimaryCtxRetain";
    case HIP_API_ID_hipDeviceGet: return "hipDeviceGet";
    case HIP_API_ID_hipStreamCreateWithFlags: return "hipStreamCreateWithFlags";
    case HIP_API_ID_hipMemcpyFromArray: return "hipMemcpyFromArray";
    case HIP_API_ID_hipMemcpy2DAsync: return "hipMemcpy2DAsync";
    case HIP_API_ID_hipFuncGetAttributes: return "hipFuncGetAttributes";
    case HIP_API_ID_hipGetSymbolSize: return "hipGetSymbolSize";
    case HIP_API_ID_hipHostFree: return "hipHostFree";
    case HIP_API_ID_hipEventCreateWithFlags: return "hipEventCreateWithFlags";
    case HIP_API_ID_hipStreamQuery: return "hipStreamQuery";
    case HIP_API_ID_hipMemcpy3D: return "hipMemcpy3D";
    case HIP_API_ID_hipMemcpyToSymbol: return "hipMemcpyToSymbol";
    case HIP_API_ID_hipMemcpy: return "hipMemcpy";
    case HIP_API_ID_hipPeekAtLastError: return "hipPeekAtLastError";
    case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice: return "hipExtLaunchMultiKernelMultiDevice";
    case HIP_API_ID_hipHostAlloc: return "hipHostAlloc";
    case HIP_API_ID_hipStreamAddCallback: return "hipStreamAddCallback";
    case HIP_API_ID_hipMemcpyToArray: return "hipMemcpyToArray";
    case HIP_API_ID_hipMemsetD32: return "hipMemsetD32";
    case HIP_API_ID_hipExtModuleLaunchKernel: return "hipExtModuleLaunchKernel";
    case HIP_API_ID_hipDeviceSynchronize: return "hipDeviceSynchronize";
    case HIP_API_ID_hipDeviceGetCacheConfig: return "hipDeviceGetCacheConfig";
    case HIP_API_ID_hipMalloc3D: return "hipMalloc3D";
    case HIP_API_ID_hipPointerGetAttributes: return "hipPointerGetAttributes";
    case HIP_API_ID_hipMemsetAsync: return "hipMemsetAsync";
    case HIP_API_ID_hipDeviceGetName: return "hipDeviceGetName";
    case HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags: return "hipModuleOccupancyMaxPotentialBlockSizeWithFlags";
    case HIP_API_ID_hipCtxPushCurrent: return "hipCtxPushCurrent";
    case HIP_API_ID_hipMemcpyPeer: return "hipMemcpyPeer";
    case HIP_API_ID_hipEventSynchronize: return "hipEventSynchronize";
    case HIP_API_ID_hipMemcpyDtoDAsync: return "hipMemcpyDtoDAsync";
    case HIP_API_ID_hipProfilerStart: return "hipProfilerStart";
    case HIP_API_ID_hipExtMallocWithFlags: return "hipExtMallocWithFlags";
    case HIP_API_ID_hipCtxEnablePeerAccess: return "hipCtxEnablePeerAccess";
    case HIP_API_ID_hipMemAllocHost: return "hipMemAllocHost";
    case HIP_API_ID_hipMemcpyDtoHAsync: return "hipMemcpyDtoHAsync";
    case HIP_API_ID_hipModuleLaunchKernel: return "hipModuleLaunchKernel";
    case HIP_API_ID_hipMemAllocPitch: return "hipMemAllocPitch";
    case HIP_API_ID_hipExtLaunchKernel: return "hipExtLaunchKernel";
    case HIP_API_ID_hipMemcpy2DFromArrayAsync: return "hipMemcpy2DFromArrayAsync";
    case HIP_API_ID_hipDeviceGetLimit: return "hipDeviceGetLimit";
    case HIP_API_ID_hipModuleLoadDataEx: return "hipModuleLoadDataEx";
    case HIP_API_ID_hipRuntimeGetVersion: return "hipRuntimeGetVersion";
    case HIP_API_ID_hipMemRangeGetAttribute: return "hipMemRangeGetAttribute";
    case HIP_API_ID_hipDeviceGetP2PAttribute: return "hipDeviceGetP2PAttribute";
    case HIP_API_ID_hipMemcpyPeerAsync: return "hipMemcpyPeerAsync";
    case HIP_API_ID_hipGetDeviceProperties: return "hipGetDeviceProperties";
    case HIP_API_ID_hipMemcpyDtoH: return "hipMemcpyDtoH";
    case HIP_API_ID_hipMemcpyWithStream: return "hipMemcpyWithStream";
    case HIP_API_ID_hipDeviceTotalMem: return "hipDeviceTotalMem";
    case HIP_API_ID_hipHostGetDevicePointer: return "hipHostGetDevicePointer";
    case HIP_API_ID_hipMemRangeGetAttributes: return "hipMemRangeGetAttributes";
    case HIP_API_ID_hipMemcpyParam2D: return "hipMemcpyParam2D";
    case HIP_API_ID_hipDevicePrimaryCtxReset: return "hipDevicePrimaryCtxReset";
    case HIP_API_ID_hipGetMipmappedArrayLevel: return "hipGetMipmappedArrayLevel";
    case HIP_API_ID_hipMemsetD32Async: return "hipMemsetD32Async";
    case HIP_API_ID_hipGetDevice: return "hipGetDevice";
    case HIP_API_ID_hipGetDeviceCount: return "hipGetDeviceCount";
    case HIP_API_ID_hipIpcOpenEventHandle: return "hipIpcOpenEventHandle";
  };
  return "unknown";
};

#include <string.h>
// Return HIP API ID by given name
static inline uint32_t hipApiIdByName(const char* name) {
  if (strcmp("hipDrvMemcpy3DAsync", name) == 0) return HIP_API_ID_hipDrvMemcpy3DAsync;
  if (strcmp("hipDeviceEnablePeerAccess", name) == 0) return HIP_API_ID_hipDeviceEnablePeerAccess;
  if (strcmp("hipFuncSetSharedMemConfig", name) == 0) return HIP_API_ID_hipFuncSetSharedMemConfig;
  if (strcmp("hipMemcpyToSymbolAsync", name) == 0) return HIP_API_ID_hipMemcpyToSymbolAsync;
  if (strcmp("hipMallocPitch", name) == 0) return HIP_API_ID_hipMallocPitch;
  if (strcmp("hipMalloc", name) == 0) return HIP_API_ID_hipMalloc;
  if (strcmp("hipMemsetD16", name) == 0) return HIP_API_ID_hipMemsetD16;
  if (strcmp("hipExtStreamGetCUMask", name) == 0) return HIP_API_ID_hipExtStreamGetCUMask;
  if (strcmp("hipEventRecord", name) == 0) return HIP_API_ID_hipEventRecord;
  if (strcmp("hipCtxSynchronize", name) == 0) return HIP_API_ID_hipCtxSynchronize;
  if (strcmp("hipSetDevice", name) == 0) return HIP_API_ID_hipSetDevice;
  if (strcmp("hipCtxGetApiVersion", name) == 0) return HIP_API_ID_hipCtxGetApiVersion;
  if (strcmp("hipMemcpyFromSymbolAsync", name) == 0) return HIP_API_ID_hipMemcpyFromSymbolAsync;
  if (strcmp("hipExtGetLinkTypeAndHopCount", name) == 0) return HIP_API_ID_hipExtGetLinkTypeAndHopCount;
  if (strcmp("__hipPopCallConfiguration", name) == 0) return HIP_API_ID___hipPopCallConfiguration;
  if (strcmp("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor", name) == 0) return HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
  if (strcmp("hipMemset3D", name) == 0) return HIP_API_ID_hipMemset3D;
  if (strcmp("hipStreamCreateWithPriority", name) == 0) return HIP_API_ID_hipStreamCreateWithPriority;
  if (strcmp("hipMemcpy2DToArray", name) == 0) return HIP_API_ID_hipMemcpy2DToArray;
  if (strcmp("hipMemsetD8Async", name) == 0) return HIP_API_ID_hipMemsetD8Async;
  if (strcmp("hipCtxGetCacheConfig", name) == 0) return HIP_API_ID_hipCtxGetCacheConfig;
  if (strcmp("hipModuleGetFunction", name) == 0) return HIP_API_ID_hipModuleGetFunction;
  if (strcmp("hipStreamWaitEvent", name) == 0) return HIP_API_ID_hipStreamWaitEvent;
  if (strcmp("hipDeviceGetStreamPriorityRange", name) == 0) return HIP_API_ID_hipDeviceGetStreamPriorityRange;
  if (strcmp("hipModuleLoad", name) == 0) return HIP_API_ID_hipModuleLoad;
  if (strcmp("hipDevicePrimaryCtxSetFlags", name) == 0) return HIP_API_ID_hipDevicePrimaryCtxSetFlags;
  if (strcmp("hipLaunchCooperativeKernel", name) == 0) return HIP_API_ID_hipLaunchCooperativeKernel;
  if (strcmp("hipLaunchCooperativeKernelMultiDevice", name) == 0) return HIP_API_ID_hipLaunchCooperativeKernelMultiDevice;
  if (strcmp("hipMemcpyAsync", name) == 0) return HIP_API_ID_hipMemcpyAsync;
  if (strcmp("hipMalloc3DArray", name) == 0) return HIP_API_ID_hipMalloc3DArray;
  if (strcmp("hipMallocHost", name) == 0) return HIP_API_ID_hipMallocHost;
  if (strcmp("hipCtxGetCurrent", name) == 0) return HIP_API_ID_hipCtxGetCurrent;
  if (strcmp("hipDevicePrimaryCtxGetState", name) == 0) return HIP_API_ID_hipDevicePrimaryCtxGetState;
  if (strcmp("hipEventQuery", name) == 0) return HIP_API_ID_hipEventQuery;
  if (strcmp("hipEventCreate", name) == 0) return HIP_API_ID_hipEventCreate;
  if (strcmp("hipMemGetAddressRange", name) == 0) return HIP_API_ID_hipMemGetAddressRange;
  if (strcmp("hipMemcpyFromSymbol", name) == 0) return HIP_API_ID_hipMemcpyFromSymbol;
  if (strcmp("hipArrayCreate", name) == 0) return HIP_API_ID_hipArrayCreate;
  if (strcmp("hipStreamAttachMemAsync", name) == 0) return HIP_API_ID_hipStreamAttachMemAsync;
  if (strcmp("hipStreamGetFlags", name) == 0) return HIP_API_ID_hipStreamGetFlags;
  if (strcmp("hipMallocArray", name) == 0) return HIP_API_ID_hipMallocArray;
  if (strcmp("hipCtxGetSharedMemConfig", name) == 0) return HIP_API_ID_hipCtxGetSharedMemConfig;
  if (strcmp("hipDeviceDisablePeerAccess", name) == 0) return HIP_API_ID_hipDeviceDisablePeerAccess;
  if (strcmp("hipModuleOccupancyMaxPotentialBlockSize", name) == 0) return HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize;
  if (strcmp("hipMemPtrGetInfo", name) == 0) return HIP_API_ID_hipMemPtrGetInfo;
  if (strcmp("hipFuncGetAttribute", name) == 0) return HIP_API_ID_hipFuncGetAttribute;
  if (strcmp("hipCtxGetFlags", name) == 0) return HIP_API_ID_hipCtxGetFlags;
  if (strcmp("hipStreamDestroy", name) == 0) return HIP_API_ID_hipStreamDestroy;
  if (strcmp("__hipPushCallConfiguration", name) == 0) return HIP_API_ID___hipPushCallConfiguration;
  if (strcmp("hipMemset3DAsync", name) == 0) return HIP_API_ID_hipMemset3DAsync;
  if (strcmp("hipDeviceGetPCIBusId", name) == 0) return HIP_API_ID_hipDeviceGetPCIBusId;
  if (strcmp("hipInit", name) == 0) return HIP_API_ID_hipInit;
  if (strcmp("hipMemcpyAtoH", name) == 0) return HIP_API_ID_hipMemcpyAtoH;
  if (strcmp("hipStreamGetPriority", name) == 0) return HIP_API_ID_hipStreamGetPriority;
  if (strcmp("hipMemset2D", name) == 0) return HIP_API_ID_hipMemset2D;
  if (strcmp("hipMemset2DAsync", name) == 0) return HIP_API_ID_hipMemset2DAsync;
  if (strcmp("hipDeviceCanAccessPeer", name) == 0) return HIP_API_ID_hipDeviceCanAccessPeer;
  if (strcmp("hipLaunchByPtr", name) == 0) return HIP_API_ID_hipLaunchByPtr;
  if (strcmp("hipMemPrefetchAsync", name) == 0) return HIP_API_ID_hipMemPrefetchAsync;
  if (strcmp("hipCtxDestroy", name) == 0) return HIP_API_ID_hipCtxDestroy;
  if (strcmp("hipMemsetD16Async", name) == 0) return HIP_API_ID_hipMemsetD16Async;
  if (strcmp("hipModuleUnload", name) == 0) return HIP_API_ID_hipModuleUnload;
  if (strcmp("hipHostUnregister", name) == 0) return HIP_API_ID_hipHostUnregister;
  if (strcmp("hipProfilerStop", name) == 0) return HIP_API_ID_hipProfilerStop;
  if (strcmp("hipExtStreamCreateWithCUMask", name) == 0) return HIP_API_ID_hipExtStreamCreateWithCUMask;
  if (strcmp("hipStreamSynchronize", name) == 0) return HIP_API_ID_hipStreamSynchronize;
  if (strcmp("hipFreeHost", name) == 0) return HIP_API_ID_hipFreeHost;
  if (strcmp("hipDeviceSetCacheConfig", name) == 0) return HIP_API_ID_hipDeviceSetCacheConfig;
  if (strcmp("hipGetErrorName", name) == 0) return HIP_API_ID_hipGetErrorName;
  if (strcmp("hipMemcpyHtoD", name) == 0) return HIP_API_ID_hipMemcpyHtoD;
  if (strcmp("hipModuleGetGlobal", name) == 0) return HIP_API_ID_hipModuleGetGlobal;
  if (strcmp("hipMemcpyHtoA", name) == 0) return HIP_API_ID_hipMemcpyHtoA;
  if (strcmp("hipCtxCreate", name) == 0) return HIP_API_ID_hipCtxCreate;
  if (strcmp("hipMemcpy2D", name) == 0) return HIP_API_ID_hipMemcpy2D;
  if (strcmp("hipIpcCloseMemHandle", name) == 0) return HIP_API_ID_hipIpcCloseMemHandle;
  if (strcmp("hipChooseDevice", name) == 0) return HIP_API_ID_hipChooseDevice;
  if (strcmp("hipDeviceSetSharedMemConfig", name) == 0) return HIP_API_ID_hipDeviceSetSharedMemConfig;
  if (strcmp("hipMallocMipmappedArray", name) == 0) return HIP_API_ID_hipMallocMipmappedArray;
  if (strcmp("hipSetupArgument", name) == 0) return HIP_API_ID_hipSetupArgument;
  if (strcmp("hipIpcGetEventHandle", name) == 0) return HIP_API_ID_hipIpcGetEventHandle;
  if (strcmp("hipFreeArray", name) == 0) return HIP_API_ID_hipFreeArray;
  if (strcmp("hipCtxSetCacheConfig", name) == 0) return HIP_API_ID_hipCtxSetCacheConfig;
  if (strcmp("hipFuncSetCacheConfig", name) == 0) return HIP_API_ID_hipFuncSetCacheConfig;
  if (strcmp("hipLaunchKernel", name) == 0) return HIP_API_ID_hipLaunchKernel;
  if (strcmp("hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", name) == 0) return HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
  if (strcmp("hipModuleGetTexRef", name) == 0) return HIP_API_ID_hipModuleGetTexRef;
  if (strcmp("hipFuncSetAttribute", name) == 0) return HIP_API_ID_hipFuncSetAttribute;
  if (strcmp("hipEventElapsedTime", name) == 0) return HIP_API_ID_hipEventElapsedTime;
  if (strcmp("hipConfigureCall", name) == 0) return HIP_API_ID_hipConfigureCall;
  if (strcmp("hipMemAdvise", name) == 0) return HIP_API_ID_hipMemAdvise;
  if (strcmp("hipMemcpy3DAsync", name) == 0) return HIP_API_ID_hipMemcpy3DAsync;
  if (strcmp("hipEventDestroy", name) == 0) return HIP_API_ID_hipEventDestroy;
  if (strcmp("hipCtxPopCurrent", name) == 0) return HIP_API_ID_hipCtxPopCurrent;
  if (strcmp("hipGetSymbolAddress", name) == 0) return HIP_API_ID_hipGetSymbolAddress;
  if (strcmp("hipHostGetFlags", name) == 0) return HIP_API_ID_hipHostGetFlags;
  if (strcmp("hipHostMalloc", name) == 0) return HIP_API_ID_hipHostMalloc;
  if (strcmp("hipCtxSetSharedMemConfig", name) == 0) return HIP_API_ID_hipCtxSetSharedMemConfig;
  if (strcmp("hipFreeMipmappedArray", name) == 0) return HIP_API_ID_hipFreeMipmappedArray;
  if (strcmp("hipMemGetInfo", name) == 0) return HIP_API_ID_hipMemGetInfo;
  if (strcmp("hipDeviceReset", name) == 0) return HIP_API_ID_hipDeviceReset;
  if (strcmp("hipMemset", name) == 0) return HIP_API_ID_hipMemset;
  if (strcmp("hipMemsetD8", name) == 0) return HIP_API_ID_hipMemsetD8;
  if (strcmp("hipMemcpyParam2DAsync", name) == 0) return HIP_API_ID_hipMemcpyParam2DAsync;
  if (strcmp("hipHostRegister", name) == 0) return HIP_API_ID_hipHostRegister;
  if (strcmp("hipDriverGetVersion", name) == 0) return HIP_API_ID_hipDriverGetVersion;
  if (strcmp("hipArray3DCreate", name) == 0) return HIP_API_ID_hipArray3DCreate;
  if (strcmp("hipIpcOpenMemHandle", name) == 0) return HIP_API_ID_hipIpcOpenMemHandle;
  if (strcmp("hipGetLastError", name) == 0) return HIP_API_ID_hipGetLastError;
  if (strcmp("hipGetDeviceFlags", name) == 0) return HIP_API_ID_hipGetDeviceFlags;
  if (strcmp("hipDeviceGetSharedMemConfig", name) == 0) return HIP_API_ID_hipDeviceGetSharedMemConfig;
  if (strcmp("hipDrvMemcpy3D", name) == 0) return HIP_API_ID_hipDrvMemcpy3D;
  if (strcmp("hipMemcpy2DFromArray", name) == 0) return HIP_API_ID_hipMemcpy2DFromArray;
  if (strcmp("hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", name) == 0) return HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
  if (strcmp("hipSetDeviceFlags", name) == 0) return HIP_API_ID_hipSetDeviceFlags;
  if (strcmp("hipHccModuleLaunchKernel", name) == 0) return HIP_API_ID_hipHccModuleLaunchKernel;
  if (strcmp("hipFree", name) == 0) return HIP_API_ID_hipFree;
  if (strcmp("hipOccupancyMaxPotentialBlockSize", name) == 0) return HIP_API_ID_hipOccupancyMaxPotentialBlockSize;
  if (strcmp("hipDeviceGetAttribute", name) == 0) return HIP_API_ID_hipDeviceGetAttribute;
  if (strcmp("hipDeviceComputeCapability", name) == 0) return HIP_API_ID_hipDeviceComputeCapability;
  if (strcmp("hipCtxDisablePeerAccess", name) == 0) return HIP_API_ID_hipCtxDisablePeerAccess;
  if (strcmp("hipMallocManaged", name) == 0) return HIP_API_ID_hipMallocManaged;
  if (strcmp("hipDeviceGetByPCIBusId", name) == 0) return HIP_API_ID_hipDeviceGetByPCIBusId;
  if (strcmp("hipIpcGetMemHandle", name) == 0) return HIP_API_ID_hipIpcGetMemHandle;
  if (strcmp("hipMemcpyHtoDAsync", name) == 0) return HIP_API_ID_hipMemcpyHtoDAsync;
  if (strcmp("hipCtxGetDevice", name) == 0) return HIP_API_ID_hipCtxGetDevice;
  if (strcmp("hipMemcpyDtoD", name) == 0) return HIP_API_ID_hipMemcpyDtoD;
  if (strcmp("hipModuleLoadData", name) == 0) return HIP_API_ID_hipModuleLoadData;
  if (strcmp("hipDevicePrimaryCtxRelease", name) == 0) return HIP_API_ID_hipDevicePrimaryCtxRelease;
  if (strcmp("hipOccupancyMaxActiveBlocksPerMultiprocessor", name) == 0) return HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor;
  if (strcmp("hipCtxSetCurrent", name) == 0) return HIP_API_ID_hipCtxSetCurrent;
  if (strcmp("hipGetErrorString", name) == 0) return HIP_API_ID_hipGetErrorString;
  if (strcmp("hipStreamCreate", name) == 0) return HIP_API_ID_hipStreamCreate;
  if (strcmp("hipDevicePrimaryCtxRetain", name) == 0) return HIP_API_ID_hipDevicePrimaryCtxRetain;
  if (strcmp("hipDeviceGet", name) == 0) return HIP_API_ID_hipDeviceGet;
  if (strcmp("hipStreamCreateWithFlags", name) == 0) return HIP_API_ID_hipStreamCreateWithFlags;
  if (strcmp("hipMemcpyFromArray", name) == 0) return HIP_API_ID_hipMemcpyFromArray;
  if (strcmp("hipMemcpy2DAsync", name) == 0) return HIP_API_ID_hipMemcpy2DAsync;
  if (strcmp("hipFuncGetAttributes", name) == 0) return HIP_API_ID_hipFuncGetAttributes;
  if (strcmp("hipGetSymbolSize", name) == 0) return HIP_API_ID_hipGetSymbolSize;
  if (strcmp("hipHostFree", name) == 0) return HIP_API_ID_hipHostFree;
  if (strcmp("hipEventCreateWithFlags", name) == 0) return HIP_API_ID_hipEventCreateWithFlags;
  if (strcmp("hipStreamQuery", name) == 0) return HIP_API_ID_hipStreamQuery;
  if (strcmp("hipMemcpy3D", name) == 0) return HIP_API_ID_hipMemcpy3D;
  if (strcmp("hipMemcpyToSymbol", name) == 0) return HIP_API_ID_hipMemcpyToSymbol;
  if (strcmp("hipMemcpy", name) == 0) return HIP_API_ID_hipMemcpy;
  if (strcmp("hipPeekAtLastError", name) == 0) return HIP_API_ID_hipPeekAtLastError;
  if (strcmp("hipExtLaunchMultiKernelMultiDevice", name) == 0) return HIP_API_ID_hipExtLaunchMultiKernelMultiDevice;
  if (strcmp("hipHostAlloc", name) == 0) return HIP_API_ID_hipHostAlloc;
  if (strcmp("hipStreamAddCallback", name) == 0) return HIP_API_ID_hipStreamAddCallback;
  if (strcmp("hipMemcpyToArray", name) == 0) return HIP_API_ID_hipMemcpyToArray;
  if (strcmp("hipMemsetD32", name) == 0) return HIP_API_ID_hipMemsetD32;
  if (strcmp("hipExtModuleLaunchKernel", name) == 0) return HIP_API_ID_hipExtModuleLaunchKernel;
  if (strcmp("hipDeviceSynchronize", name) == 0) return HIP_API_ID_hipDeviceSynchronize;
  if (strcmp("hipDeviceGetCacheConfig", name) == 0) return HIP_API_ID_hipDeviceGetCacheConfig;
  if (strcmp("hipMalloc3D", name) == 0) return HIP_API_ID_hipMalloc3D;
  if (strcmp("hipPointerGetAttributes", name) == 0) return HIP_API_ID_hipPointerGetAttributes;
  if (strcmp("hipMemsetAsync", name) == 0) return HIP_API_ID_hipMemsetAsync;
  if (strcmp("hipDeviceGetName", name) == 0) return HIP_API_ID_hipDeviceGetName;
  if (strcmp("hipModuleOccupancyMaxPotentialBlockSizeWithFlags", name) == 0) return HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
  if (strcmp("hipCtxPushCurrent", name) == 0) return HIP_API_ID_hipCtxPushCurrent;
  if (strcmp("hipMemcpyPeer", name) == 0) return HIP_API_ID_hipMemcpyPeer;
  if (strcmp("hipEventSynchronize", name) == 0) return HIP_API_ID_hipEventSynchronize;
  if (strcmp("hipMemcpyDtoDAsync", name) == 0) return HIP_API_ID_hipMemcpyDtoDAsync;
  if (strcmp("hipProfilerStart", name) == 0) return HIP_API_ID_hipProfilerStart;
  if (strcmp("hipExtMallocWithFlags", name) == 0) return HIP_API_ID_hipExtMallocWithFlags;
  if (strcmp("hipCtxEnablePeerAccess", name) == 0) return HIP_API_ID_hipCtxEnablePeerAccess;
  if (strcmp("hipMemAllocHost", name) == 0) return HIP_API_ID_hipMemAllocHost;
  if (strcmp("hipMemcpyDtoHAsync", name) == 0) return HIP_API_ID_hipMemcpyDtoHAsync;
  if (strcmp("hipModuleLaunchKernel", name) == 0) return HIP_API_ID_hipModuleLaunchKernel;
  if (strcmp("hipMemAllocPitch", name) == 0) return HIP_API_ID_hipMemAllocPitch;
  if (strcmp("hipExtLaunchKernel", name) == 0) return HIP_API_ID_hipExtLaunchKernel;
  if (strcmp("hipMemcpy2DFromArrayAsync", name) == 0) return HIP_API_ID_hipMemcpy2DFromArrayAsync;
  if (strcmp("hipDeviceGetLimit", name) == 0) return HIP_API_ID_hipDeviceGetLimit;
  if (strcmp("hipModuleLoadDataEx", name) == 0) return HIP_API_ID_hipModuleLoadDataEx;
  if (strcmp("hipRuntimeGetVersion", name) == 0) return HIP_API_ID_hipRuntimeGetVersion;
  if (strcmp("hipMemRangeGetAttribute", name) == 0) return HIP_API_ID_hipMemRangeGetAttribute;
  if (strcmp("hipDeviceGetP2PAttribute", name) == 0) return HIP_API_ID_hipDeviceGetP2PAttribute;
  if (strcmp("hipMemcpyPeerAsync", name) == 0) return HIP_API_ID_hipMemcpyPeerAsync;
  if (strcmp("hipGetDeviceProperties", name) == 0) return HIP_API_ID_hipGetDeviceProperties;
  if (strcmp("hipMemcpyDtoH", name) == 0) return HIP_API_ID_hipMemcpyDtoH;
  if (strcmp("hipMemcpyWithStream", name) == 0) return HIP_API_ID_hipMemcpyWithStream;
  if (strcmp("hipDeviceTotalMem", name) == 0) return HIP_API_ID_hipDeviceTotalMem;
  if (strcmp("hipHostGetDevicePointer", name) == 0) return HIP_API_ID_hipHostGetDevicePointer;
  if (strcmp("hipMemRangeGetAttributes", name) == 0) return HIP_API_ID_hipMemRangeGetAttributes;
  if (strcmp("hipMemcpyParam2D", name) == 0) return HIP_API_ID_hipMemcpyParam2D;
  if (strcmp("hipDevicePrimaryCtxReset", name) == 0) return HIP_API_ID_hipDevicePrimaryCtxReset;
  if (strcmp("hipGetMipmappedArrayLevel", name) == 0) return HIP_API_ID_hipGetMipmappedArrayLevel;
  if (strcmp("hipMemsetD32Async", name) == 0) return HIP_API_ID_hipMemsetD32Async;
  if (strcmp("hipGetDevice", name) == 0) return HIP_API_ID_hipGetDevice;
  if (strcmp("hipGetDeviceCount", name) == 0) return HIP_API_ID_hipGetDeviceCount;
  if (strcmp("hipIpcOpenEventHandle", name) == 0) return HIP_API_ID_hipIpcOpenEventHandle;
  return HIP_API_ID_NUMBER;
}

// HIP API callbacks data structure
typedef struct hip_api_data_s {
  uint64_t correlation_id;
  uint32_t phase;
  union {
    struct {
      const HIP_MEMCPY3D* pCopy;
      HIP_MEMCPY3D pCopy__val;
      hipStream_t stream;
    } hipDrvMemcpy3DAsync;
    struct {
      int peerDeviceId;
      unsigned int flags;
    } hipDeviceEnablePeerAccess;
    struct {
      const void* func;
      hipSharedMemConfig config;
    } hipFuncSetSharedMemConfig;
    struct {
      const void* symbol;
      const void* src;
      size_t sizeBytes;
      size_t offset;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpyToSymbolAsync;
    struct {
      void** ptr;
      void* ptr__val;
      size_t* pitch;
      size_t pitch__val;
      size_t width;
      size_t height;
    } hipMallocPitch;
    struct {
      void** ptr;
      void* ptr__val;
      size_t size;
    } hipMalloc;
    struct {
      hipDeviceptr_t dest;
      unsigned short value;
      size_t count;
    } hipMemsetD16;
    struct {
      hipStream_t stream;
      unsigned int cuMaskSize;
      unsigned int* cuMask;
      unsigned int cuMask__val;
    } hipExtStreamGetCUMask;
    struct {
      hipEvent_t event;
      hipStream_t stream;
    } hipEventRecord;
    struct {
      int deviceId;
    } hipSetDevice;
    struct {
      hipCtx_t ctx;
      int* apiVersion;
      int apiVersion__val;
    } hipCtxGetApiVersion;
    struct {
      void* dst;
      const void* symbol;
      size_t sizeBytes;
      size_t offset;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpyFromSymbolAsync;
    struct {
      int device1;
      int device2;
      unsigned int* linktype;
      unsigned int linktype__val;
      unsigned int* hopcount;
      unsigned int hopcount__val;
    } hipExtGetLinkTypeAndHopCount;
    struct {
      dim3* gridDim;
      dim3 gridDim__val;
      dim3* blockDim;
      dim3 blockDim__val;
      size_t* sharedMem;
      size_t sharedMem__val;
      hipStream_t* stream;
      hipStream_t stream__val;
    } __hipPopCallConfiguration;
    struct {
      int* numBlocks;
      int numBlocks__val;
      hipFunction_t f;
      int blockSize;
      size_t dynSharedMemPerBlk;
    } hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
    struct {
      hipPitchedPtr pitchedDevPtr;
      int value;
      hipExtent extent;
    } hipMemset3D;
    struct {
      hipStream_t* stream;
      hipStream_t stream__val;
      unsigned int flags;
      int priority;
    } hipStreamCreateWithPriority;
    struct {
      hipArray* dst;
      hipArray dst__val;
      size_t wOffset;
      size_t hOffset;
      const void* src;
      size_t spitch;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
    } hipMemcpy2DToArray;
    struct {
      hipDeviceptr_t dest;
      unsigned char value;
      size_t count;
      hipStream_t stream;
    } hipMemsetD8Async;
    struct {
      hipFuncCache_t* cacheConfig;
      hipFuncCache_t cacheConfig__val;
    } hipCtxGetCacheConfig;
    struct {
      hipFunction_t* function;
      hipFunction_t function__val;
      hipModule_t module;
      const char* kname;
      char kname__val;
    } hipModuleGetFunction;
    struct {
      hipStream_t stream;
      hipEvent_t event;
      unsigned int flags;
    } hipStreamWaitEvent;
    struct {
      int* leastPriority;
      int leastPriority__val;
      int* greatestPriority;
      int greatestPriority__val;
    } hipDeviceGetStreamPriorityRange;
    struct {
      hipModule_t* module;
      hipModule_t module__val;
      const char* fname;
      char fname__val;
    } hipModuleLoad;
    struct {
      hipDevice_t dev;
      unsigned int flags;
    } hipDevicePrimaryCtxSetFlags;
    struct {
      const void* f;
      dim3 gridDim;
      dim3 blockDimX;
      void** kernelParams;
      void* kernelParams__val;
      unsigned int sharedMemBytes;
      hipStream_t stream;
    } hipLaunchCooperativeKernel;
    struct {
      hipLaunchParams* launchParamsList;
      hipLaunchParams launchParamsList__val;
      int numDevices;
      unsigned int flags;
    } hipLaunchCooperativeKernelMultiDevice;
    struct {
      void* dst;
      const void* src;
      size_t sizeBytes;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpyAsync;
    struct {
      hipArray_t* array;
      hipArray_t array__val;
      const hipChannelFormatDesc* desc;
      hipChannelFormatDesc desc__val;
      hipExtent extent;
      unsigned int flags;
    } hipMalloc3DArray;
    struct {
      void** ptr;
      void* ptr__val;
      size_t size;
    } hipMallocHost;
    struct {
      hipCtx_t* ctx;
      hipCtx_t ctx__val;
    } hipCtxGetCurrent;
    struct {
      hipDevice_t dev;
      unsigned int* flags;
      unsigned int flags__val;
      int* active;
      int active__val;
    } hipDevicePrimaryCtxGetState;
    struct {
      hipEvent_t event;
    } hipEventQuery;
    struct {
      hipEvent_t* event;
      hipEvent_t event__val;
    } hipEventCreate;
    struct {
      hipDeviceptr_t* pbase;
      hipDeviceptr_t pbase__val;
      size_t* psize;
      size_t psize__val;
      hipDeviceptr_t dptr;
    } hipMemGetAddressRange;
    struct {
      void* dst;
      const void* symbol;
      size_t sizeBytes;
      size_t offset;
      hipMemcpyKind kind;
    } hipMemcpyFromSymbol;
    struct {
      hipArray** pHandle;
      hipArray* pHandle__val;
      const HIP_ARRAY_DESCRIPTOR* pAllocateArray;
      HIP_ARRAY_DESCRIPTOR pAllocateArray__val;
    } hipArrayCreate;
    struct {
      hipStream_t stream;
      hipDeviceptr_t* dev_ptr;
      hipDeviceptr_t dev_ptr__val;
      size_t length;
      unsigned int flags;
    } hipStreamAttachMemAsync;
    struct {
      hipStream_t stream;
      unsigned int* flags;
      unsigned int flags__val;
    } hipStreamGetFlags;
    struct {
      hipArray** array;
      hipArray* array__val;
      const hipChannelFormatDesc* desc;
      hipChannelFormatDesc desc__val;
      size_t width;
      size_t height;
      unsigned int flags;
    } hipMallocArray;
    struct {
      hipSharedMemConfig* pConfig;
      hipSharedMemConfig pConfig__val;
    } hipCtxGetSharedMemConfig;
    struct {
      int peerDeviceId;
    } hipDeviceDisablePeerAccess;
    struct {
      int* gridSize;
      int gridSize__val;
      int* blockSize;
      int blockSize__val;
      hipFunction_t f;
      size_t dynSharedMemPerBlk;
      int blockSizeLimit;
    } hipModuleOccupancyMaxPotentialBlockSize;
    struct {
      void* ptr;
      size_t* size;
      size_t size__val;
    } hipMemPtrGetInfo;
    struct {
      int* value;
      int value__val;
      hipFunction_attribute attrib;
      hipFunction_t hfunc;
    } hipFuncGetAttribute;
    struct {
      unsigned int* flags;
      unsigned int flags__val;
    } hipCtxGetFlags;
    struct {
      hipStream_t stream;
    } hipStreamDestroy;
    struct {
      dim3 gridDim;
      dim3 blockDim;
      size_t sharedMem;
      hipStream_t stream;
    } __hipPushCallConfiguration;
    struct {
      hipPitchedPtr pitchedDevPtr;
      int value;
      hipExtent extent;
      hipStream_t stream;
    } hipMemset3DAsync;
    struct {
      char* pciBusId;
      char pciBusId__val;
      int len;
      int device;
    } hipDeviceGetPCIBusId;
    struct {
      unsigned int flags;
    } hipInit;
    struct {
      void* dst;
      hipArray* srcArray;
      hipArray srcArray__val;
      size_t srcOffset;
      size_t count;
    } hipMemcpyAtoH;
    struct {
      hipStream_t stream;
      int* priority;
      int priority__val;
    } hipStreamGetPriority;
    struct {
      void* dst;
      size_t pitch;
      int value;
      size_t width;
      size_t height;
    } hipMemset2D;
    struct {
      void* dst;
      size_t pitch;
      int value;
      size_t width;
      size_t height;
      hipStream_t stream;
    } hipMemset2DAsync;
    struct {
      int* canAccessPeer;
      int canAccessPeer__val;
      int deviceId;
      int peerDeviceId;
    } hipDeviceCanAccessPeer;
    struct {
      const void* hostFunction;
    } hipLaunchByPtr;
    struct {
      const void* dev_ptr;
      size_t count;
      int device;
      hipStream_t stream;
    } hipMemPrefetchAsync;
    struct {
      hipCtx_t ctx;
    } hipCtxDestroy;
    struct {
      hipDeviceptr_t dest;
      unsigned short value;
      size_t count;
      hipStream_t stream;
    } hipMemsetD16Async;
    struct {
      hipModule_t module;
    } hipModuleUnload;
    struct {
      void* hostPtr;
    } hipHostUnregister;
    struct {
      hipStream_t* stream;
      hipStream_t stream__val;
      unsigned int cuMaskSize;
      const unsigned int* cuMask;
      unsigned int cuMask__val;
    } hipExtStreamCreateWithCUMask;
    struct {
      hipStream_t stream;
    } hipStreamSynchronize;
    struct {
      void* ptr;
    } hipFreeHost;
    struct {
      hipFuncCache_t cacheConfig;
    } hipDeviceSetCacheConfig;
    struct {
      hipDeviceptr_t dst;
      void* src;
      size_t sizeBytes;
    } hipMemcpyHtoD;
    struct {
      hipDeviceptr_t* dptr;
      hipDeviceptr_t dptr__val;
      size_t* bytes;
      size_t bytes__val;
      hipModule_t hmod;
      const char* name;
      char name__val;
    } hipModuleGetGlobal;
    struct {
      hipArray* dstArray;
      hipArray dstArray__val;
      size_t dstOffset;
      const void* srcHost;
      size_t count;
    } hipMemcpyHtoA;
    struct {
      hipCtx_t* ctx;
      hipCtx_t ctx__val;
      unsigned int flags;
      hipDevice_t device;
    } hipCtxCreate;
    struct {
      void* dst;
      size_t dpitch;
      const void* src;
      size_t spitch;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
    } hipMemcpy2D;
    struct {
      void* devPtr;
    } hipIpcCloseMemHandle;
    struct {
      int* device;
      int device__val;
      const hipDeviceProp_t* prop;
      hipDeviceProp_t prop__val;
    } hipChooseDevice;
    struct {
      hipSharedMemConfig config;
    } hipDeviceSetSharedMemConfig;
    struct {
      hipMipmappedArray_t* mipmappedArray;
      hipMipmappedArray_t mipmappedArray__val;
      const hipChannelFormatDesc* desc;
      hipChannelFormatDesc desc__val;
      hipExtent extent;
      unsigned int numLevels;
      unsigned int flags;
    } hipMallocMipmappedArray;
    struct {
      const void* arg;
      size_t size;
      size_t offset;
    } hipSetupArgument;
    struct {
      hipIpcEventHandle_t* handle;
      hipIpcEventHandle_t handle__val;
      hipEvent_t event;
    } hipIpcGetEventHandle;
    struct {
      hipArray* array;
      hipArray array__val;
    } hipFreeArray;
    struct {
      hipFuncCache_t cacheConfig;
    } hipCtxSetCacheConfig;
    struct {
      const void* func;
      hipFuncCache_t config;
    } hipFuncSetCacheConfig;
    struct {
      const void* function_address;
      dim3 numBlocks;
      dim3 dimBlocks;
      void** args;
      void* args__val;
      size_t sharedMemBytes;
      hipStream_t stream;
    } hipLaunchKernel;
    struct {
      int* numBlocks;
      int numBlocks__val;
      hipFunction_t f;
      int blockSize;
      size_t dynSharedMemPerBlk;
      unsigned int flags;
    } hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
    struct {
      textureReference** texRef;
      textureReference* texRef__val;
      hipModule_t hmod;
      const char* name;
      char name__val;
    } hipModuleGetTexRef;
    struct {
      const void* func;
      hipFuncAttribute attr;
      int value;
    } hipFuncSetAttribute;
    struct {
      float* ms;
      float ms__val;
      hipEvent_t start;
      hipEvent_t stop;
    } hipEventElapsedTime;
    struct {
      dim3 gridDim;
      dim3 blockDim;
      size_t sharedMem;
      hipStream_t stream;
    } hipConfigureCall;
    struct {
      const void* dev_ptr;
      size_t count;
      hipMemoryAdvise advice;
      int device;
    } hipMemAdvise;
    struct {
      const hipMemcpy3DParms* p;
      hipMemcpy3DParms p__val;
      hipStream_t stream;
    } hipMemcpy3DAsync;
    struct {
      hipEvent_t event;
    } hipEventDestroy;
    struct {
      hipCtx_t* ctx;
      hipCtx_t ctx__val;
    } hipCtxPopCurrent;
    struct {
      void** devPtr;
      void* devPtr__val;
      const void* symbol;
    } hipGetSymbolAddress;
    struct {
      unsigned int* flagsPtr;
      unsigned int flagsPtr__val;
      void* hostPtr;
    } hipHostGetFlags;
    struct {
      void** ptr;
      void* ptr__val;
      size_t size;
      unsigned int flags;
    } hipHostMalloc;
    struct {
      hipSharedMemConfig config;
    } hipCtxSetSharedMemConfig;
    struct {
      hipMipmappedArray_t mipmappedArray;
    } hipFreeMipmappedArray;
    struct {
      size_t* free;
      size_t free__val;
      size_t* total;
      size_t total__val;
    } hipMemGetInfo;
    struct {
      void* dst;
      int value;
      size_t sizeBytes;
    } hipMemset;
    struct {
      hipDeviceptr_t dest;
      unsigned char value;
      size_t count;
    } hipMemsetD8;
    struct {
      const hip_Memcpy2D* pCopy;
      hip_Memcpy2D pCopy__val;
      hipStream_t stream;
    } hipMemcpyParam2DAsync;
    struct {
      void* hostPtr;
      size_t sizeBytes;
      unsigned int flags;
    } hipHostRegister;
    struct {
      int* driverVersion;
      int driverVersion__val;
    } hipDriverGetVersion;
    struct {
      hipArray** array;
      hipArray* array__val;
      const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray;
      HIP_ARRAY3D_DESCRIPTOR pAllocateArray__val;
    } hipArray3DCreate;
    struct {
      void** devPtr;
      void* devPtr__val;
      hipIpcMemHandle_t handle;
      unsigned int flags;
    } hipIpcOpenMemHandle;
    struct {
      unsigned int* flags;
      unsigned int flags__val;
    } hipGetDeviceFlags;
    struct {
      hipSharedMemConfig* pConfig;
      hipSharedMemConfig pConfig__val;
    } hipDeviceGetSharedMemConfig;
    struct {
      const HIP_MEMCPY3D* pCopy;
      HIP_MEMCPY3D pCopy__val;
    } hipDrvMemcpy3D;
    struct {
      void* dst;
      size_t dpitch;
      hipArray_const_t src;
      size_t wOffset;
      size_t hOffset;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
    } hipMemcpy2DFromArray;
    struct {
      int* numBlocks;
      int numBlocks__val;
      const void* f;
      int blockSize;
      size_t dynamicSMemSize;
      unsigned int flags;
    } hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
    struct {
      unsigned int flags;
    } hipSetDeviceFlags;
    struct {
      hipFunction_t f;
      unsigned int globalWorkSizeX;
      unsigned int globalWorkSizeY;
      unsigned int globalWorkSizeZ;
      unsigned int blockDimX;
      unsigned int blockDimY;
      unsigned int blockDimZ;
      size_t sharedMemBytes;
      hipStream_t hStream;
      void** kernelParams;
      void* kernelParams__val;
      void** extra;
      void* extra__val;
      hipEvent_t startEvent;
      hipEvent_t stopEvent;
    } hipHccModuleLaunchKernel;
    struct {
      void* ptr;
    } hipFree;
    struct {
      int* gridSize;
      int gridSize__val;
      int* blockSize;
      int blockSize__val;
      const void* f;
      size_t dynSharedMemPerBlk;
      int blockSizeLimit;
    } hipOccupancyMaxPotentialBlockSize;
    struct {
      int* pi;
      int pi__val;
      hipDeviceAttribute_t attr;
      int deviceId;
    } hipDeviceGetAttribute;
    struct {
      int* major;
      int major__val;
      int* minor;
      int minor__val;
      hipDevice_t device;
    } hipDeviceComputeCapability;
    struct {
      hipCtx_t peerCtx;
    } hipCtxDisablePeerAccess;
    struct {
      void** dev_ptr;
      void* dev_ptr__val;
      size_t size;
      unsigned int flags;
    } hipMallocManaged;
    struct {
      int* device;
      int device__val;
      const char* pciBusId;
      char pciBusId__val;
    } hipDeviceGetByPCIBusId;
    struct {
      hipIpcMemHandle_t* handle;
      hipIpcMemHandle_t handle__val;
      void* devPtr;
    } hipIpcGetMemHandle;
    struct {
      hipDeviceptr_t dst;
      void* src;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyHtoDAsync;
    struct {
      hipDevice_t* device;
      hipDevice_t device__val;
    } hipCtxGetDevice;
    struct {
      hipDeviceptr_t dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
    } hipMemcpyDtoD;
    struct {
      hipModule_t* module;
      hipModule_t module__val;
      const void* image;
    } hipModuleLoadData;
    struct {
      hipDevice_t dev;
    } hipDevicePrimaryCtxRelease;
    struct {
      int* numBlocks;
      int numBlocks__val;
      const void* f;
      int blockSize;
      size_t dynamicSMemSize;
    } hipOccupancyMaxActiveBlocksPerMultiprocessor;
    struct {
      hipCtx_t ctx;
    } hipCtxSetCurrent;
    struct {
      hipStream_t* stream;
      hipStream_t stream__val;
    } hipStreamCreate;
    struct {
      hipCtx_t* pctx;
      hipCtx_t pctx__val;
      hipDevice_t dev;
    } hipDevicePrimaryCtxRetain;
    struct {
      hipDevice_t* device;
      hipDevice_t device__val;
      int ordinal;
    } hipDeviceGet;
    struct {
      hipStream_t* stream;
      hipStream_t stream__val;
      unsigned int flags;
    } hipStreamCreateWithFlags;
    struct {
      void* dst;
      hipArray_const_t srcArray;
      size_t wOffset;
      size_t hOffset;
      size_t count;
      hipMemcpyKind kind;
    } hipMemcpyFromArray;
    struct {
      void* dst;
      size_t dpitch;
      const void* src;
      size_t spitch;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpy2DAsync;
    struct {
      hipFuncAttributes* attr;
      hipFuncAttributes attr__val;
      const void* func;
    } hipFuncGetAttributes;
    struct {
      size_t* size;
      size_t size__val;
      const void* symbol;
    } hipGetSymbolSize;
    struct {
      void* ptr;
    } hipHostFree;
    struct {
      hipEvent_t* event;
      hipEvent_t event__val;
      unsigned int flags;
    } hipEventCreateWithFlags;
    struct {
      hipStream_t stream;
    } hipStreamQuery;
    struct {
      const hipMemcpy3DParms* p;
      hipMemcpy3DParms p__val;
    } hipMemcpy3D;
    struct {
      const void* symbol;
      const void* src;
      size_t sizeBytes;
      size_t offset;
      hipMemcpyKind kind;
    } hipMemcpyToSymbol;
    struct {
      void* dst;
      const void* src;
      size_t sizeBytes;
      hipMemcpyKind kind;
    } hipMemcpy;
    struct {
      hipLaunchParams* launchParamsList;
      hipLaunchParams launchParamsList__val;
      int numDevices;
      unsigned int flags;
    } hipExtLaunchMultiKernelMultiDevice;
    struct {
      void** ptr;
      void* ptr__val;
      size_t size;
      unsigned int flags;
    } hipHostAlloc;
    struct {
      hipStream_t stream;
      hipStreamCallback_t callback;
      void* userData;
      unsigned int flags;
    } hipStreamAddCallback;
    struct {
      hipArray* dst;
      hipArray dst__val;
      size_t wOffset;
      size_t hOffset;
      const void* src;
      size_t count;
      hipMemcpyKind kind;
    } hipMemcpyToArray;
    struct {
      hipDeviceptr_t dest;
      int value;
      size_t count;
    } hipMemsetD32;
    struct {
      hipFunction_t f;
      unsigned int globalWorkSizeX;
      unsigned int globalWorkSizeY;
      unsigned int globalWorkSizeZ;
      unsigned int localWorkSizeX;
      unsigned int localWorkSizeY;
      unsigned int localWorkSizeZ;
      size_t sharedMemBytes;
      hipStream_t hStream;
      void** kernelParams;
      void* kernelParams__val;
      void** extra;
      void* extra__val;
      hipEvent_t startEvent;
      hipEvent_t stopEvent;
      unsigned int flags;
    } hipExtModuleLaunchKernel;
    struct {
      hipFuncCache_t* cacheConfig;
      hipFuncCache_t cacheConfig__val;
    } hipDeviceGetCacheConfig;
    struct {
      hipPitchedPtr* pitchedDevPtr;
      hipPitchedPtr pitchedDevPtr__val;
      hipExtent extent;
    } hipMalloc3D;
    struct {
      hipPointerAttribute_t* attributes;
      hipPointerAttribute_t attributes__val;
      const void* ptr;
    } hipPointerGetAttributes;
    struct {
      void* dst;
      int value;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemsetAsync;
    struct {
      char* name;
      char name__val;
      int len;
      hipDevice_t device;
    } hipDeviceGetName;
    struct {
      int* gridSize;
      int gridSize__val;
      int* blockSize;
      int blockSize__val;
      hipFunction_t f;
      size_t dynSharedMemPerBlk;
      int blockSizeLimit;
      unsigned int flags;
    } hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
    struct {
      hipCtx_t ctx;
    } hipCtxPushCurrent;
    struct {
      void* dst;
      int dstDeviceId;
      const void* src;
      int srcDeviceId;
      size_t sizeBytes;
    } hipMemcpyPeer;
    struct {
      hipEvent_t event;
    } hipEventSynchronize;
    struct {
      hipDeviceptr_t dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyDtoDAsync;
    struct {
      void** ptr;
      void* ptr__val;
      size_t sizeBytes;
      unsigned int flags;
    } hipExtMallocWithFlags;
    struct {
      hipCtx_t peerCtx;
      unsigned int flags;
    } hipCtxEnablePeerAccess;
    struct {
      void** ptr;
      void* ptr__val;
      size_t size;
    } hipMemAllocHost;
    struct {
      void* dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyDtoHAsync;
    struct {
      hipFunction_t f;
      unsigned int gridDimX;
      unsigned int gridDimY;
      unsigned int gridDimZ;
      unsigned int blockDimX;
      unsigned int blockDimY;
      unsigned int blockDimZ;
      unsigned int sharedMemBytes;
      hipStream_t stream;
      void** kernelParams;
      void* kernelParams__val;
      void** extra;
      void* extra__val;
    } hipModuleLaunchKernel;
    struct {
      hipDeviceptr_t* dptr;
      hipDeviceptr_t dptr__val;
      size_t* pitch;
      size_t pitch__val;
      size_t widthInBytes;
      size_t height;
      unsigned int elementSizeBytes;
    } hipMemAllocPitch;
    struct {
      const void* function_address;
      dim3 numBlocks;
      dim3 dimBlocks;
      void** args;
      void* args__val;
      size_t sharedMemBytes;
      hipStream_t stream;
      hipEvent_t startEvent;
      hipEvent_t stopEvent;
      int flags;
    } hipExtLaunchKernel;
    struct {
      void* dst;
      size_t dpitch;
      hipArray_const_t src;
      size_t wOffset;
      size_t hOffset;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpy2DFromArrayAsync;
    struct {
      size_t* pValue;
      size_t pValue__val;
      enum hipLimit_t limit;
    } hipDeviceGetLimit;
    struct {
      hipModule_t* module;
      hipModule_t module__val;
      const void* image;
      unsigned int numOptions;
      hipJitOption* options;
      hipJitOption options__val;
      void** optionsValues;
      void* optionsValues__val;
    } hipModuleLoadDataEx;
    struct {
      int* runtimeVersion;
      int runtimeVersion__val;
    } hipRuntimeGetVersion;
    struct {
      void* data;
      size_t data_size;
      hipMemRangeAttribute attribute;
      const void* dev_ptr;
      size_t count;
    } hipMemRangeGetAttribute;
    struct {
      int* value;
      int value__val;
      hipDeviceP2PAttr attr;
      int srcDevice;
      int dstDevice;
    } hipDeviceGetP2PAttribute;
    struct {
      void* dst;
      int dstDeviceId;
      const void* src;
      int srcDevice;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyPeerAsync;
    struct {
      hipDeviceProp_t* props;
      hipDeviceProp_t props__val;
      hipDevice_t device;
    } hipGetDeviceProperties;
    struct {
      void* dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
    } hipMemcpyDtoH;
    struct {
      void* dst;
      const void* src;
      size_t sizeBytes;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpyWithStream;
    struct {
      size_t* bytes;
      size_t bytes__val;
      hipDevice_t device;
    } hipDeviceTotalMem;
    struct {
      void** devPtr;
      void* devPtr__val;
      void* hstPtr;
      unsigned int flags;
    } hipHostGetDevicePointer;
    struct {
      void** data;
      void* data__val;
      size_t* data_sizes;
      size_t data_sizes__val;
      hipMemRangeAttribute* attributes;
      hipMemRangeAttribute attributes__val;
      size_t num_attributes;
      const void* dev_ptr;
      size_t count;
    } hipMemRangeGetAttributes;
    struct {
      const hip_Memcpy2D* pCopy;
      hip_Memcpy2D pCopy__val;
    } hipMemcpyParam2D;
    struct {
      hipDevice_t dev;
    } hipDevicePrimaryCtxReset;
    struct {
      hipArray_t* levelArray;
      hipArray_t levelArray__val;
      hipMipmappedArray_const_t mipmappedArray;
      unsigned int level;
    } hipGetMipmappedArrayLevel;
    struct {
      hipDeviceptr_t dst;
      int value;
      size_t count;
      hipStream_t stream;
    } hipMemsetD32Async;
    struct {
      int* deviceId;
      int deviceId__val;
    } hipGetDevice;
    struct {
      int* count;
      int count__val;
    } hipGetDeviceCount;
    struct {
      hipEvent_t* event;
      hipEvent_t event__val;
      hipIpcEventHandle_t handle;
    } hipIpcOpenEventHandle;
  } args;
} hip_api_data_t;

// HIP API callbacks args data filling macros
// hipDrvMemcpy3DAsync[('const HIP_MEMCPY3D*', 'pCopy'), ('hipStream_t', 'stream')]
#define INIT_hipDrvMemcpy3DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDrvMemcpy3DAsync.pCopy = (const HIP_MEMCPY3D*)pCopy; \
  cb_data.args.hipDrvMemcpy3DAsync.stream = (hipStream_t)stream; \
};
// hipDeviceEnablePeerAccess[('int', 'peerDeviceId'), ('unsigned int', 'flags')]
#define INIT_hipDeviceEnablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceEnablePeerAccess.peerDeviceId = (int)peerDeviceId; \
  cb_data.args.hipDeviceEnablePeerAccess.flags = (unsigned int)flags; \
};
// hipFuncSetSharedMemConfig[('const void*', 'func'), ('hipSharedMemConfig', 'config')]
#define INIT_hipFuncSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncSetSharedMemConfig.func = (const void*)func; \
  cb_data.args.hipFuncSetSharedMemConfig.config = (hipSharedMemConfig)config; \
};
// hipMemcpyToSymbolAsync[('const void*', 'symbol'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyToSymbolAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyToSymbolAsync.symbol = (const void*)symbol; \
  cb_data.args.hipMemcpyToSymbolAsync.src = (const void*)src; \
  cb_data.args.hipMemcpyToSymbolAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyToSymbolAsync.offset = (size_t)offset; \
  cb_data.args.hipMemcpyToSymbolAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpyToSymbolAsync.stream = (hipStream_t)stream; \
};
// hipMallocPitch[('void**', 'ptr'), ('size_t*', 'pitch'), ('size_t', 'width'), ('size_t', 'height')]
#define INIT_hipMallocPitch_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocPitch.ptr = (void**)ptr; \
  cb_data.args.hipMallocPitch.pitch = (size_t*)pitch; \
  cb_data.args.hipMallocPitch.width = (size_t)width; \
  cb_data.args.hipMallocPitch.height = (size_t)height; \
};
// hipMalloc[('void**', 'ptr'), ('size_t', 'size')]
#define INIT_hipMalloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc.ptr = (void**)ptr; \
  cb_data.args.hipMalloc.size = (size_t)sizeBytes; \
};
// hipMemsetD16[('hipDeviceptr_t', 'dest'), ('unsigned short', 'value'), ('size_t', 'count')]
#define INIT_hipMemsetD16_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD16.dest = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemsetD16.value = (unsigned short)value; \
  cb_data.args.hipMemsetD16.count = (size_t)count; \
};
// hipExtStreamGetCUMask[('hipStream_t', 'stream'), ('unsigned int', 'cuMaskSize'), ('unsigned int*', 'cuMask')]
#define INIT_hipExtStreamGetCUMask_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtStreamGetCUMask.stream = (hipStream_t)stream; \
  cb_data.args.hipExtStreamGetCUMask.cuMaskSize = (unsigned int)cuMaskSize; \
  cb_data.args.hipExtStreamGetCUMask.cuMask = (unsigned int*)cuMask; \
};
// hipEventRecord[('hipEvent_t', 'event'), ('hipStream_t', 'stream')]
#define INIT_hipEventRecord_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventRecord.event = (hipEvent_t)event; \
  cb_data.args.hipEventRecord.stream = (hipStream_t)stream; \
};
// hipCtxSynchronize[]
#define INIT_hipCtxSynchronize_CB_ARGS_DATA(cb_data) { \
};
// hipSetDevice[('int', 'deviceId')]
#define INIT_hipSetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetDevice.deviceId = (int)device; \
};
// hipCtxGetApiVersion[('hipCtx_t', 'ctx'), ('int*', 'apiVersion')]
#define INIT_hipCtxGetApiVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetApiVersion.ctx = (hipCtx_t)ctx; \
  cb_data.args.hipCtxGetApiVersion.apiVersion = (int*)apiVersion; \
};
// hipMemcpyFromSymbolAsync[('void*', 'dst'), ('const void*', 'symbol'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyFromSymbolAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyFromSymbolAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpyFromSymbolAsync.symbol = (const void*)symbol; \
  cb_data.args.hipMemcpyFromSymbolAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyFromSymbolAsync.offset = (size_t)offset; \
  cb_data.args.hipMemcpyFromSymbolAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpyFromSymbolAsync.stream = (hipStream_t)stream; \
};
// hipExtGetLinkTypeAndHopCount[('int', 'device1'), ('int', 'device2'), ('unsigned int*', 'linktype'), ('unsigned int*', 'hopcount')]
#define INIT_hipExtGetLinkTypeAndHopCount_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtGetLinkTypeAndHopCount.device1 = (int)device1; \
  cb_data.args.hipExtGetLinkTypeAndHopCount.device2 = (int)device2; \
  cb_data.args.hipExtGetLinkTypeAndHopCount.linktype = (unsigned int*)linktype; \
  cb_data.args.hipExtGetLinkTypeAndHopCount.hopcount = (unsigned int*)hopcount; \
};
// __hipPopCallConfiguration[('dim3*', 'gridDim'), ('dim3*', 'blockDim'), ('size_t*', 'sharedMem'), ('hipStream_t*', 'stream')]
#define INIT___hipPopCallConfiguration_CB_ARGS_DATA(cb_data) { \
  cb_data.args.__hipPopCallConfiguration.gridDim = (dim3*)gridDim; \
  cb_data.args.__hipPopCallConfiguration.blockDim = (dim3*)blockDim; \
  cb_data.args.__hipPopCallConfiguration.sharedMem = (size_t*)sharedMem; \
  cb_data.args.__hipPopCallConfiguration.stream = (hipStream_t*)stream; \
};
// hipModuleOccupancyMaxActiveBlocksPerMultiprocessor[('int*', 'numBlocks'), ('hipFunction_t', 'f'), ('int', 'blockSize'), ('size_t', 'dynSharedMemPerBlk')]
#define INIT_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks = (int*)numBlocks; \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.f = (hipFunction_t)f; \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.blockSize = (int)blockSize; \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.dynSharedMemPerBlk = (size_t)dynSharedMemPerBlk; \
};
// hipMemset3D[('hipPitchedPtr', 'pitchedDevPtr'), ('int', 'value'), ('hipExtent', 'extent')]
#define INIT_hipMemset3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset3D.pitchedDevPtr = (hipPitchedPtr)pitchedDevPtr; \
  cb_data.args.hipMemset3D.value = (int)value; \
  cb_data.args.hipMemset3D.extent = (hipExtent)extent; \
};
// hipStreamCreateWithPriority[('hipStream_t*', 'stream'), ('unsigned int', 'flags'), ('int', 'priority')]
#define INIT_hipStreamCreateWithPriority_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreateWithPriority.stream = (hipStream_t*)stream; \
  cb_data.args.hipStreamCreateWithPriority.flags = (unsigned int)flags; \
  cb_data.args.hipStreamCreateWithPriority.priority = (int)priority; \
};
// hipMemcpy2DToArray[('hipArray*', 'dst'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpy2DToArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DToArray.dst = (hipArray*)dst; \
  cb_data.args.hipMemcpy2DToArray.wOffset = (size_t)wOffset; \
  cb_data.args.hipMemcpy2DToArray.hOffset = (size_t)hOffset; \
  cb_data.args.hipMemcpy2DToArray.src = (const void*)src; \
  cb_data.args.hipMemcpy2DToArray.spitch = (size_t)spitch; \
  cb_data.args.hipMemcpy2DToArray.width = (size_t)width; \
  cb_data.args.hipMemcpy2DToArray.height = (size_t)height; \
  cb_data.args.hipMemcpy2DToArray.kind = (hipMemcpyKind)kind; \
};
// hipMemsetD8Async[('hipDeviceptr_t', 'dest'), ('unsigned char', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
#define INIT_hipMemsetD8Async_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD8Async.dest = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemsetD8Async.value = (unsigned char)value; \
  cb_data.args.hipMemsetD8Async.count = (size_t)count; \
  cb_data.args.hipMemsetD8Async.stream = (hipStream_t)stream; \
};
// hipCtxGetCacheConfig[('hipFuncCache_t*', 'cacheConfig')]
#define INIT_hipCtxGetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetCacheConfig.cacheConfig = (hipFuncCache_t*)cacheConfig; \
};
// hipModuleGetFunction[('hipFunction_t*', 'function'), ('hipModule_t', 'module'), ('const char*', 'kname')]
#define INIT_hipModuleGetFunction_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetFunction.function = (hipFunction_t*)hfunc; \
  cb_data.args.hipModuleGetFunction.module = (hipModule_t)hmod; \
  cb_data.args.hipModuleGetFunction.kname = (name) ? strdup(name) : NULL; \
};
// hipStreamWaitEvent[('hipStream_t', 'stream'), ('hipEvent_t', 'event'), ('unsigned int', 'flags')]
#define INIT_hipStreamWaitEvent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamWaitEvent.stream = (hipStream_t)stream; \
  cb_data.args.hipStreamWaitEvent.event = (hipEvent_t)event; \
  cb_data.args.hipStreamWaitEvent.flags = (unsigned int)flags; \
};
// hipDeviceGetStreamPriorityRange[('int*', 'leastPriority'), ('int*', 'greatestPriority')]
#define INIT_hipDeviceGetStreamPriorityRange_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetStreamPriorityRange.leastPriority = (int*)leastPriority; \
  cb_data.args.hipDeviceGetStreamPriorityRange.greatestPriority = (int*)greatestPriority; \
};
// hipModuleLoad[('hipModule_t*', 'module'), ('const char*', 'fname')]
#define INIT_hipModuleLoad_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoad.module = (hipModule_t*)module; \
  cb_data.args.hipModuleLoad.fname = (fname) ? strdup(fname) : NULL; \
};
// hipDevicePrimaryCtxSetFlags[('hipDevice_t', 'dev'), ('unsigned int', 'flags')]
#define INIT_hipDevicePrimaryCtxSetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxSetFlags.dev = (hipDevice_t)dev; \
  cb_data.args.hipDevicePrimaryCtxSetFlags.flags = (unsigned int)flags; \
};
// hipLaunchCooperativeKernel[('const void*', 'f'), ('dim3', 'gridDim'), ('dim3', 'blockDimX'), ('void**', 'kernelParams'), ('unsigned int', 'sharedMemBytes'), ('hipStream_t', 'stream')]
#define INIT_hipLaunchCooperativeKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipLaunchCooperativeKernel.f = (const void*)f; \
  cb_data.args.hipLaunchCooperativeKernel.gridDim = (dim3)gridDim; \
  cb_data.args.hipLaunchCooperativeKernel.blockDimX = (dim3)blockDim; \
  cb_data.args.hipLaunchCooperativeKernel.kernelParams = (void**)kernelParams; \
  cb_data.args.hipLaunchCooperativeKernel.sharedMemBytes = (unsigned int)sharedMemBytes; \
  cb_data.args.hipLaunchCooperativeKernel.stream = (hipStream_t)hStream; \
};
// hipLaunchCooperativeKernelMultiDevice[('hipLaunchParams*', 'launchParamsList'), ('int', 'numDevices'), ('unsigned int', 'flags')]
#define INIT_hipLaunchCooperativeKernelMultiDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipLaunchCooperativeKernelMultiDevice.launchParamsList = (hipLaunchParams*)launchParamsList; \
  cb_data.args.hipLaunchCooperativeKernelMultiDevice.numDevices = (int)numDevices; \
  cb_data.args.hipLaunchCooperativeKernelMultiDevice.flags = (unsigned int)flags; \
};
// hipMemcpyAsync[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpyAsync.src = (const void*)src; \
  cb_data.args.hipMemcpyAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpyAsync.stream = (hipStream_t)stream; \
};
// hipMalloc3DArray[('hipArray_t*', 'array'), ('const hipChannelFormatDesc*', 'desc'), ('hipExtent', 'extent'), ('unsigned int', 'flags')]
#define INIT_hipMalloc3DArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc3DArray.array = (hipArray_t*)array; \
  cb_data.args.hipMalloc3DArray.desc = (const hipChannelFormatDesc*)desc; \
  cb_data.args.hipMalloc3DArray.extent = (hipExtent)extent; \
  cb_data.args.hipMalloc3DArray.flags = (unsigned int)flags; \
};
// hipMallocHost[('void**', 'ptr'), ('size_t', 'size')]
#define INIT_hipMallocHost_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocHost.ptr = (void**)ptr; \
  cb_data.args.hipMallocHost.size = (size_t)size; \
};
// hipCtxGetCurrent[('hipCtx_t*', 'ctx')]
#define INIT_hipCtxGetCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetCurrent.ctx = (hipCtx_t*)ctx; \
};
// hipDevicePrimaryCtxGetState[('hipDevice_t', 'dev'), ('unsigned int*', 'flags'), ('int*', 'active')]
#define INIT_hipDevicePrimaryCtxGetState_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxGetState.dev = (hipDevice_t)dev; \
  cb_data.args.hipDevicePrimaryCtxGetState.flags = (unsigned int*)flags; \
  cb_data.args.hipDevicePrimaryCtxGetState.active = (int*)active; \
};
// hipEventQuery[('hipEvent_t', 'event')]
#define INIT_hipEventQuery_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventQuery.event = (hipEvent_t)event; \
};
// hipEventCreate[('hipEvent_t*', 'event')]
#define INIT_hipEventCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventCreate.event = (hipEvent_t*)event; \
};
// hipMemGetAddressRange[('hipDeviceptr_t*', 'pbase'), ('size_t*', 'psize'), ('hipDeviceptr_t', 'dptr')]
#define INIT_hipMemGetAddressRange_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemGetAddressRange.pbase = (hipDeviceptr_t*)pbase; \
  cb_data.args.hipMemGetAddressRange.psize = (size_t*)psize; \
  cb_data.args.hipMemGetAddressRange.dptr = (hipDeviceptr_t)dptr; \
};
// hipMemcpyFromSymbol[('void*', 'dst'), ('const void*', 'symbol'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpyFromSymbol_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyFromSymbol.dst = (void*)dst; \
  cb_data.args.hipMemcpyFromSymbol.symbol = (const void*)symbol; \
  cb_data.args.hipMemcpyFromSymbol.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyFromSymbol.offset = (size_t)offset; \
  cb_data.args.hipMemcpyFromSymbol.kind = (hipMemcpyKind)kind; \
};
// hipArrayCreate[('hipArray**', 'pHandle'), ('const HIP_ARRAY_DESCRIPTOR*', 'pAllocateArray')]
#define INIT_hipArrayCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipArrayCreate.pHandle = (hipArray**)array; \
  cb_data.args.hipArrayCreate.pAllocateArray = (const HIP_ARRAY_DESCRIPTOR*)pAllocateArray; \
};
// hipStreamAttachMemAsync[('hipStream_t', 'stream'), ('hipDeviceptr_t*', 'dev_ptr'), ('size_t', 'length'), ('unsigned int', 'flags')]
#define INIT_hipStreamAttachMemAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamAttachMemAsync.stream = (hipStream_t)stream; \
  cb_data.args.hipStreamAttachMemAsync.dev_ptr = (hipDeviceptr_t*)dev_ptr; \
  cb_data.args.hipStreamAttachMemAsync.length = (size_t)length; \
  cb_data.args.hipStreamAttachMemAsync.flags = (unsigned int)flags; \
};
// hipStreamGetFlags[('hipStream_t', 'stream'), ('unsigned int*', 'flags')]
#define INIT_hipStreamGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamGetFlags.stream = (hipStream_t)stream; \
  cb_data.args.hipStreamGetFlags.flags = (unsigned int*)flags; \
};
// hipMallocArray[('hipArray**', 'array'), ('const hipChannelFormatDesc*', 'desc'), ('size_t', 'width'), ('size_t', 'height'), ('unsigned int', 'flags')]
#define INIT_hipMallocArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocArray.array = (hipArray**)array; \
  cb_data.args.hipMallocArray.desc = (const hipChannelFormatDesc*)desc; \
  cb_data.args.hipMallocArray.width = (size_t)width; \
  cb_data.args.hipMallocArray.height = (size_t)height; \
  cb_data.args.hipMallocArray.flags = (unsigned int)flags; \
};
// hipCtxGetSharedMemConfig[('hipSharedMemConfig*', 'pConfig')]
#define INIT_hipCtxGetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetSharedMemConfig.pConfig = (hipSharedMemConfig*)pConfig; \
};
// hipDeviceDisablePeerAccess[('int', 'peerDeviceId')]
#define INIT_hipDeviceDisablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceDisablePeerAccess.peerDeviceId = (int)peerDeviceId; \
};
// hipModuleOccupancyMaxPotentialBlockSize[('int*', 'gridSize'), ('int*', 'blockSize'), ('hipFunction_t', 'f'), ('size_t', 'dynSharedMemPerBlk'), ('int', 'blockSizeLimit')]
#define INIT_hipModuleOccupancyMaxPotentialBlockSize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSize.gridSize = (int*)gridSize; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSize.blockSize = (int*)blockSize; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSize.f = (hipFunction_t)f; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSize.dynSharedMemPerBlk = (size_t)dynSharedMemPerBlk; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSize.blockSizeLimit = (int)blockSizeLimit; \
};
// hipMemPtrGetInfo[('void*', 'ptr'), ('size_t*', 'size')]
#define INIT_hipMemPtrGetInfo_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemPtrGetInfo.ptr = (void*)ptr; \
  cb_data.args.hipMemPtrGetInfo.size = (size_t*)size; \
};
// hipFuncGetAttribute[('int*', 'value'), ('hipFunction_attribute', 'attrib'), ('hipFunction_t', 'hfunc')]
#define INIT_hipFuncGetAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncGetAttribute.value = (int*)value; \
  cb_data.args.hipFuncGetAttribute.attrib = (hipFunction_attribute)attrib; \
  cb_data.args.hipFuncGetAttribute.hfunc = (hipFunction_t)hfunc; \
};
// hipCtxGetFlags[('unsigned int*', 'flags')]
#define INIT_hipCtxGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetFlags.flags = (unsigned int*)flags; \
};
// hipStreamDestroy[('hipStream_t', 'stream')]
#define INIT_hipStreamDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamDestroy.stream = (hipStream_t)stream; \
};
// __hipPushCallConfiguration[('dim3', 'gridDim'), ('dim3', 'blockDim'), ('size_t', 'sharedMem'), ('hipStream_t', 'stream')]
#define INIT___hipPushCallConfiguration_CB_ARGS_DATA(cb_data) { \
  cb_data.args.__hipPushCallConfiguration.gridDim = (dim3)gridDim; \
  cb_data.args.__hipPushCallConfiguration.blockDim = (dim3)blockDim; \
  cb_data.args.__hipPushCallConfiguration.sharedMem = (size_t)sharedMem; \
  cb_data.args.__hipPushCallConfiguration.stream = (hipStream_t)stream; \
};
// hipMemset3DAsync[('hipPitchedPtr', 'pitchedDevPtr'), ('int', 'value'), ('hipExtent', 'extent'), ('hipStream_t', 'stream')]
#define INIT_hipMemset3DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset3DAsync.pitchedDevPtr = (hipPitchedPtr)pitchedDevPtr; \
  cb_data.args.hipMemset3DAsync.value = (int)value; \
  cb_data.args.hipMemset3DAsync.extent = (hipExtent)extent; \
  cb_data.args.hipMemset3DAsync.stream = (hipStream_t)stream; \
};
// hipDeviceGetPCIBusId[('char*', 'pciBusId'), ('int', 'len'), ('int', 'device')]
#define INIT_hipDeviceGetPCIBusId_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetPCIBusId.pciBusId = (char*)pciBusId; \
  cb_data.args.hipDeviceGetPCIBusId.len = (int)len; \
  cb_data.args.hipDeviceGetPCIBusId.device = (int)device; \
};
// hipInit[('unsigned int', 'flags')]
#define INIT_hipInit_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipInit.flags = (unsigned int)flags; \
};
// hipMemcpyAtoH[('void*', 'dst'), ('hipArray*', 'srcArray'), ('size_t', 'srcOffset'), ('size_t', 'count')]
#define INIT_hipMemcpyAtoH_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyAtoH.dst = (void*)dstHost; \
  cb_data.args.hipMemcpyAtoH.srcArray = (hipArray*)srcArray; \
  cb_data.args.hipMemcpyAtoH.srcOffset = (size_t)srcOffset; \
  cb_data.args.hipMemcpyAtoH.count = (size_t)ByteCount; \
};
// hipStreamGetPriority[('hipStream_t', 'stream'), ('int*', 'priority')]
#define INIT_hipStreamGetPriority_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamGetPriority.stream = (hipStream_t)stream; \
  cb_data.args.hipStreamGetPriority.priority = (int*)priority; \
};
// hipMemset2D[('void*', 'dst'), ('size_t', 'pitch'), ('int', 'value'), ('size_t', 'width'), ('size_t', 'height')]
#define INIT_hipMemset2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset2D.dst = (void*)dst; \
  cb_data.args.hipMemset2D.pitch = (size_t)pitch; \
  cb_data.args.hipMemset2D.value = (int)value; \
  cb_data.args.hipMemset2D.width = (size_t)width; \
  cb_data.args.hipMemset2D.height = (size_t)height; \
};
// hipMemset2DAsync[('void*', 'dst'), ('size_t', 'pitch'), ('int', 'value'), ('size_t', 'width'), ('size_t', 'height'), ('hipStream_t', 'stream')]
#define INIT_hipMemset2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset2DAsync.dst = (void*)dst; \
  cb_data.args.hipMemset2DAsync.pitch = (size_t)pitch; \
  cb_data.args.hipMemset2DAsync.value = (int)value; \
  cb_data.args.hipMemset2DAsync.width = (size_t)width; \
  cb_data.args.hipMemset2DAsync.height = (size_t)height; \
  cb_data.args.hipMemset2DAsync.stream = (hipStream_t)stream; \
};
// hipDeviceCanAccessPeer[('int*', 'canAccessPeer'), ('int', 'deviceId'), ('int', 'peerDeviceId')]
#define INIT_hipDeviceCanAccessPeer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceCanAccessPeer.canAccessPeer = (int*)canAccess; \
  cb_data.args.hipDeviceCanAccessPeer.deviceId = (int)deviceId; \
  cb_data.args.hipDeviceCanAccessPeer.peerDeviceId = (int)peerDeviceId; \
};
// hipLaunchByPtr[('const void*', 'hostFunction')]
#define INIT_hipLaunchByPtr_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipLaunchByPtr.hostFunction = (const void*)hostFunction; \
};
// hipMemPrefetchAsync[('const void*', 'dev_ptr'), ('size_t', 'count'), ('int', 'device'), ('hipStream_t', 'stream')]
#define INIT_hipMemPrefetchAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemPrefetchAsync.dev_ptr = (const void*)dev_ptr; \
  cb_data.args.hipMemPrefetchAsync.count = (size_t)count; \
  cb_data.args.hipMemPrefetchAsync.device = (int)device; \
  cb_data.args.hipMemPrefetchAsync.stream = (hipStream_t)stream; \
};
// hipCtxDestroy[('hipCtx_t', 'ctx')]
#define INIT_hipCtxDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxDestroy.ctx = (hipCtx_t)ctx; \
};
// hipMemsetD16Async[('hipDeviceptr_t', 'dest'), ('unsigned short', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
#define INIT_hipMemsetD16Async_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD16Async.dest = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemsetD16Async.value = (unsigned short)value; \
  cb_data.args.hipMemsetD16Async.count = (size_t)count; \
  cb_data.args.hipMemsetD16Async.stream = (hipStream_t)stream; \
};
// hipModuleUnload[('hipModule_t', 'module')]
#define INIT_hipModuleUnload_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleUnload.module = (hipModule_t)hmod; \
};
// hipHostUnregister[('void*', 'hostPtr')]
#define INIT_hipHostUnregister_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostUnregister.hostPtr = (void*)hostPtr; \
};
// hipProfilerStop[]
#define INIT_hipProfilerStop_CB_ARGS_DATA(cb_data) { \
};
// hipExtStreamCreateWithCUMask[('hipStream_t*', 'stream'), ('unsigned int', 'cuMaskSize'), ('const unsigned int*', 'cuMask')]
#define INIT_hipExtStreamCreateWithCUMask_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtStreamCreateWithCUMask.stream = (hipStream_t*)stream; \
  cb_data.args.hipExtStreamCreateWithCUMask.cuMaskSize = (unsigned int)cuMaskSize; \
  cb_data.args.hipExtStreamCreateWithCUMask.cuMask = (const unsigned int*)cuMask; \
};
// hipStreamSynchronize[('hipStream_t', 'stream')]
#define INIT_hipStreamSynchronize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamSynchronize.stream = (hipStream_t)stream; \
};
// hipFreeHost[('void*', 'ptr')]
#define INIT_hipFreeHost_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFreeHost.ptr = (void*)ptr; \
};
// hipDeviceSetCacheConfig[('hipFuncCache_t', 'cacheConfig')]
#define INIT_hipDeviceSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceSetCacheConfig.cacheConfig = (hipFuncCache_t)cacheConfig; \
};
// hipGetErrorName[]
#define INIT_hipGetErrorName_CB_ARGS_DATA(cb_data) { \
};
// hipMemcpyHtoD[('hipDeviceptr_t', 'dst'), ('void*', 'src'), ('size_t', 'sizeBytes')]
#define INIT_hipMemcpyHtoD_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoD.dst = (hipDeviceptr_t)dstDevice; \
  cb_data.args.hipMemcpyHtoD.src = (void*)srcHost; \
  cb_data.args.hipMemcpyHtoD.sizeBytes = (size_t)ByteCount; \
};
// hipModuleGetGlobal[('hipDeviceptr_t*', 'dptr'), ('size_t*', 'bytes'), ('hipModule_t', 'hmod'), ('const char*', 'name')]
#define INIT_hipModuleGetGlobal_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetGlobal.dptr = (hipDeviceptr_t*)dptr; \
  cb_data.args.hipModuleGetGlobal.bytes = (size_t*)bytes; \
  cb_data.args.hipModuleGetGlobal.hmod = (hipModule_t)hmod; \
  cb_data.args.hipModuleGetGlobal.name = (name) ? strdup(name) : NULL; \
};
// hipMemcpyHtoA[('hipArray*', 'dstArray'), ('size_t', 'dstOffset'), ('const void*', 'srcHost'), ('size_t', 'count')]
#define INIT_hipMemcpyHtoA_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoA.dstArray = (hipArray*)dstArray; \
  cb_data.args.hipMemcpyHtoA.dstOffset = (size_t)dstOffset; \
  cb_data.args.hipMemcpyHtoA.srcHost = (const void*)srcHost; \
  cb_data.args.hipMemcpyHtoA.count = (size_t)ByteCount; \
};
// hipCtxCreate[('hipCtx_t*', 'ctx'), ('unsigned int', 'flags'), ('hipDevice_t', 'device')]
#define INIT_hipCtxCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxCreate.ctx = (hipCtx_t*)ctx; \
  cb_data.args.hipCtxCreate.flags = (unsigned int)flags; \
  cb_data.args.hipCtxCreate.device = (hipDevice_t)device; \
};
// hipMemcpy2D[('void*', 'dst'), ('size_t', 'dpitch'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpy2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2D.dst = (void*)dst; \
  cb_data.args.hipMemcpy2D.dpitch = (size_t)dpitch; \
  cb_data.args.hipMemcpy2D.src = (const void*)src; \
  cb_data.args.hipMemcpy2D.spitch = (size_t)spitch; \
  cb_data.args.hipMemcpy2D.width = (size_t)width; \
  cb_data.args.hipMemcpy2D.height = (size_t)height; \
  cb_data.args.hipMemcpy2D.kind = (hipMemcpyKind)kind; \
};
// hipIpcCloseMemHandle[('void*', 'devPtr')]
#define INIT_hipIpcCloseMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcCloseMemHandle.devPtr = (void*)dev_ptr; \
};
// hipChooseDevice[('int*', 'device'), ('const hipDeviceProp_t*', 'prop')]
#define INIT_hipChooseDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipChooseDevice.device = (int*)device; \
  cb_data.args.hipChooseDevice.prop = (const hipDeviceProp_t*)properties; \
};
// hipDeviceSetSharedMemConfig[('hipSharedMemConfig', 'config')]
#define INIT_hipDeviceSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceSetSharedMemConfig.config = (hipSharedMemConfig)config; \
};
// hipMallocMipmappedArray[('hipMipmappedArray_t*', 'mipmappedArray'), ('const hipChannelFormatDesc*', 'desc'), ('hipExtent', 'extent'), ('unsigned int', 'numLevels'), ('unsigned int', 'flags')]
#define INIT_hipMallocMipmappedArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocMipmappedArray.mipmappedArray = (hipMipmappedArray_t*)mipmappedArray; \
  cb_data.args.hipMallocMipmappedArray.desc = (const hipChannelFormatDesc*)desc; \
  cb_data.args.hipMallocMipmappedArray.extent = (hipExtent)extent; \
  cb_data.args.hipMallocMipmappedArray.numLevels = (unsigned int)numLevels; \
  cb_data.args.hipMallocMipmappedArray.flags = (unsigned int)flags; \
};
// hipSetupArgument[('const void*', 'arg'), ('size_t', 'size'), ('size_t', 'offset')]
#define INIT_hipSetupArgument_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetupArgument.arg = (const void*)arg; \
  cb_data.args.hipSetupArgument.size = (size_t)size; \
  cb_data.args.hipSetupArgument.offset = (size_t)offset; \
};
// hipIpcGetEventHandle[('hipIpcEventHandle_t*', 'handle'), ('hipEvent_t', 'event')]
#define INIT_hipIpcGetEventHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcGetEventHandle.handle = (hipIpcEventHandle_t*)handle; \
  cb_data.args.hipIpcGetEventHandle.event = (hipEvent_t)event; \
};
// hipFreeArray[('hipArray*', 'array')]
#define INIT_hipFreeArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFreeArray.array = (hipArray*)array; \
};
// hipCtxSetCacheConfig[('hipFuncCache_t', 'cacheConfig')]
#define INIT_hipCtxSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetCacheConfig.cacheConfig = (hipFuncCache_t)cacheConfig; \
};
// hipFuncSetCacheConfig[('const void*', 'func'), ('hipFuncCache_t', 'config')]
#define INIT_hipFuncSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncSetCacheConfig.func = (const void*)func; \
  cb_data.args.hipFuncSetCacheConfig.config = (hipFuncCache_t)cacheConfig; \
};
// hipLaunchKernel[('const void*', 'function_address'), ('dim3', 'numBlocks'), ('dim3', 'dimBlocks'), ('void**', 'args'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'stream')]
#define INIT_hipLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipLaunchKernel.function_address = (const void*)hostFunction; \
  cb_data.args.hipLaunchKernel.numBlocks = (dim3)gridDim; \
  cb_data.args.hipLaunchKernel.dimBlocks = (dim3)blockDim; \
  cb_data.args.hipLaunchKernel.args = (void**)args; \
  cb_data.args.hipLaunchKernel.sharedMemBytes = (size_t)sharedMemBytes; \
  cb_data.args.hipLaunchKernel.stream = (hipStream_t)stream; \
};
// hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags[('int*', 'numBlocks'), ('hipFunction_t', 'f'), ('int', 'blockSize'), ('size_t', 'dynSharedMemPerBlk'), ('unsigned int', 'flags')]
#define INIT_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks = (int*)numBlocks; \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.f = (hipFunction_t)f; \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.blockSize = (int)blockSize; \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.dynSharedMemPerBlk = (size_t)dynSharedMemPerBlk; \
  cb_data.args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.flags = (unsigned int)flags; \
};
// hipModuleGetTexRef[('textureReference**', 'texRef'), ('hipModule_t', 'hmod'), ('const char*', 'name')]
#define INIT_hipModuleGetTexRef_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetTexRef.texRef = (textureReference**)texRef; \
  cb_data.args.hipModuleGetTexRef.hmod = (hipModule_t)hmod; \
  cb_data.args.hipModuleGetTexRef.name = (name) ? strdup(name) : NULL; \
};
// hipFuncSetAttribute[('const void*', 'func'), ('hipFuncAttribute', 'attr'), ('int', 'value')]
#define INIT_hipFuncSetAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncSetAttribute.func = (const void*)func; \
  cb_data.args.hipFuncSetAttribute.attr = (hipFuncAttribute)attr; \
  cb_data.args.hipFuncSetAttribute.value = (int)value; \
};
// hipEventElapsedTime[('float*', 'ms'), ('hipEvent_t', 'start'), ('hipEvent_t', 'stop')]
#define INIT_hipEventElapsedTime_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventElapsedTime.ms = (float*)ms; \
  cb_data.args.hipEventElapsedTime.start = (hipEvent_t)start; \
  cb_data.args.hipEventElapsedTime.stop = (hipEvent_t)stop; \
};
// hipConfigureCall[('dim3', 'gridDim'), ('dim3', 'blockDim'), ('size_t', 'sharedMem'), ('hipStream_t', 'stream')]
#define INIT_hipConfigureCall_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipConfigureCall.gridDim = (dim3)gridDim; \
  cb_data.args.hipConfigureCall.blockDim = (dim3)blockDim; \
  cb_data.args.hipConfigureCall.sharedMem = (size_t)sharedMem; \
  cb_data.args.hipConfigureCall.stream = (hipStream_t)stream; \
};
// hipMemAdvise[('const void*', 'dev_ptr'), ('size_t', 'count'), ('hipMemoryAdvise', 'advice'), ('int', 'device')]
#define INIT_hipMemAdvise_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemAdvise.dev_ptr = (const void*)dev_ptr; \
  cb_data.args.hipMemAdvise.count = (size_t)count; \
  cb_data.args.hipMemAdvise.advice = (hipMemoryAdvise)advice; \
  cb_data.args.hipMemAdvise.device = (int)device; \
};
// hipMemcpy3DAsync[('const hipMemcpy3DParms*', 'p'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpy3DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy3DAsync.p = (const hipMemcpy3DParms*)p; \
  cb_data.args.hipMemcpy3DAsync.stream = (hipStream_t)stream; \
};
// hipEventDestroy[('hipEvent_t', 'event')]
#define INIT_hipEventDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventDestroy.event = (hipEvent_t)event; \
};
// hipCtxPopCurrent[('hipCtx_t*', 'ctx')]
#define INIT_hipCtxPopCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxPopCurrent.ctx = (hipCtx_t*)ctx; \
};
// hipGetSymbolAddress[('void**', 'devPtr'), ('const void*', 'symbol')]
#define INIT_hipGetSymbolAddress_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetSymbolAddress.devPtr = (void**)devPtr; \
  cb_data.args.hipGetSymbolAddress.symbol = (const void*)symbol; \
};
// hipHostGetFlags[('unsigned int*', 'flagsPtr'), ('void*', 'hostPtr')]
#define INIT_hipHostGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostGetFlags.flagsPtr = (unsigned int*)flagsPtr; \
  cb_data.args.hipHostGetFlags.hostPtr = (void*)hostPtr; \
};
// hipHostMalloc[('void**', 'ptr'), ('size_t', 'size'), ('unsigned int', 'flags')]
#define INIT_hipHostMalloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostMalloc.ptr = (void**)ptr; \
  cb_data.args.hipHostMalloc.size = (size_t)sizeBytes; \
  cb_data.args.hipHostMalloc.flags = (unsigned int)flags; \
};
// hipCtxSetSharedMemConfig[('hipSharedMemConfig', 'config')]
#define INIT_hipCtxSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetSharedMemConfig.config = (hipSharedMemConfig)config; \
};
// hipFreeMipmappedArray[('hipMipmappedArray_t', 'mipmappedArray')]
#define INIT_hipFreeMipmappedArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFreeMipmappedArray.mipmappedArray = (hipMipmappedArray_t)mipmappedArray; \
};
// hipMemGetInfo[('size_t*', 'free'), ('size_t*', 'total')]
#define INIT_hipMemGetInfo_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemGetInfo.free = (size_t*)free; \
  cb_data.args.hipMemGetInfo.total = (size_t*)total; \
};
// hipDeviceReset[]
#define INIT_hipDeviceReset_CB_ARGS_DATA(cb_data) { \
};
// hipMemset[('void*', 'dst'), ('int', 'value'), ('size_t', 'sizeBytes')]
#define INIT_hipMemset_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset.dst = (void*)dst; \
  cb_data.args.hipMemset.value = (int)value; \
  cb_data.args.hipMemset.sizeBytes = (size_t)sizeBytes; \
};
// hipMemsetD8[('hipDeviceptr_t', 'dest'), ('unsigned char', 'value'), ('size_t', 'count')]
#define INIT_hipMemsetD8_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD8.dest = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemsetD8.value = (unsigned char)value; \
  cb_data.args.hipMemsetD8.count = (size_t)count; \
};
// hipMemcpyParam2DAsync[('const hip_Memcpy2D*', 'pCopy'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyParam2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyParam2DAsync.pCopy = (const hip_Memcpy2D*)pCopy; \
  cb_data.args.hipMemcpyParam2DAsync.stream = (hipStream_t)stream; \
};
// hipHostRegister[('void*', 'hostPtr'), ('size_t', 'sizeBytes'), ('unsigned int', 'flags')]
#define INIT_hipHostRegister_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostRegister.hostPtr = (void*)hostPtr; \
  cb_data.args.hipHostRegister.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipHostRegister.flags = (unsigned int)flags; \
};
// hipDriverGetVersion[('int*', 'driverVersion')]
#define INIT_hipDriverGetVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDriverGetVersion.driverVersion = (int*)driverVersion; \
};
// hipArray3DCreate[('hipArray**', 'array'), ('const HIP_ARRAY3D_DESCRIPTOR*', 'pAllocateArray')]
#define INIT_hipArray3DCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipArray3DCreate.array = (hipArray**)array; \
  cb_data.args.hipArray3DCreate.pAllocateArray = (const HIP_ARRAY3D_DESCRIPTOR*)pAllocateArray; \
};
// hipIpcOpenMemHandle[('void**', 'devPtr'), ('hipIpcMemHandle_t', 'handle'), ('unsigned int', 'flags')]
#define INIT_hipIpcOpenMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcOpenMemHandle.devPtr = (void**)dev_ptr; \
  cb_data.args.hipIpcOpenMemHandle.handle = (hipIpcMemHandle_t)handle; \
  cb_data.args.hipIpcOpenMemHandle.flags = (unsigned int)flags; \
};
// hipGetLastError[]
#define INIT_hipGetLastError_CB_ARGS_DATA(cb_data) { \
};
// hipGetDeviceFlags[('unsigned int*', 'flags')]
#define INIT_hipGetDeviceFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceFlags.flags = (unsigned int*)flags; \
};
// hipDeviceGetSharedMemConfig[('hipSharedMemConfig*', 'pConfig')]
#define INIT_hipDeviceGetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetSharedMemConfig.pConfig = (hipSharedMemConfig*)pConfig; \
};
// hipDrvMemcpy3D[('const HIP_MEMCPY3D*', 'pCopy')]
#define INIT_hipDrvMemcpy3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDrvMemcpy3D.pCopy = (const HIP_MEMCPY3D*)pCopy; \
};
// hipMemcpy2DFromArray[('void*', 'dst'), ('size_t', 'dpitch'), ('hipArray_const_t', 'src'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpy2DFromArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DFromArray.dst = (void*)dst; \
  cb_data.args.hipMemcpy2DFromArray.dpitch = (size_t)dpitch; \
  cb_data.args.hipMemcpy2DFromArray.src = (hipArray_const_t)src; \
  cb_data.args.hipMemcpy2DFromArray.wOffset = (size_t)wOffsetSrc; \
  cb_data.args.hipMemcpy2DFromArray.hOffset = (size_t)hOffset; \
  cb_data.args.hipMemcpy2DFromArray.width = (size_t)width; \
  cb_data.args.hipMemcpy2DFromArray.height = (size_t)height; \
  cb_data.args.hipMemcpy2DFromArray.kind = (hipMemcpyKind)kind; \
};
// hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags[('int*', 'numBlocks'), ('const void*', 'f'), ('int', 'blockSize'), ('size_t', 'dynamicSMemSize'), ('unsigned int', 'flags')]
#define INIT_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks = (int*)numBlocks; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.f = (const void*)f; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.blockSize = (int)blockSize; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.dynamicSMemSize = (size_t)dynamicSMemSize; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.flags = (unsigned int)flags; \
};
// hipSetDeviceFlags[('unsigned int', 'flags')]
#define INIT_hipSetDeviceFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetDeviceFlags.flags = (unsigned int)flags; \
};
// hipHccModuleLaunchKernel[('hipFunction_t', 'f'), ('unsigned int', 'globalWorkSizeX'), ('unsigned int', 'globalWorkSizeY'), ('unsigned int', 'globalWorkSizeZ'), ('unsigned int', 'blockDimX'), ('unsigned int', 'blockDimY'), ('unsigned int', 'blockDimZ'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'hStream'), ('void**', 'kernelParams'), ('void**', 'extra'), ('hipEvent_t', 'startEvent'), ('hipEvent_t', 'stopEvent')]
#define INIT_hipHccModuleLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHccModuleLaunchKernel.f = (hipFunction_t)f; \
  cb_data.args.hipHccModuleLaunchKernel.globalWorkSizeX = (unsigned int)globalWorkSizeX; \
  cb_data.args.hipHccModuleLaunchKernel.globalWorkSizeY = (unsigned int)globalWorkSizeY; \
  cb_data.args.hipHccModuleLaunchKernel.globalWorkSizeZ = (unsigned int)globalWorkSizeZ; \
  cb_data.args.hipHccModuleLaunchKernel.blockDimX = (unsigned int)blockDimX; \
  cb_data.args.hipHccModuleLaunchKernel.blockDimY = (unsigned int)blockDimY; \
  cb_data.args.hipHccModuleLaunchKernel.blockDimZ = (unsigned int)blockDimZ; \
  cb_data.args.hipHccModuleLaunchKernel.sharedMemBytes = (size_t)sharedMemBytes; \
  cb_data.args.hipHccModuleLaunchKernel.hStream = (hipStream_t)hStream; \
  cb_data.args.hipHccModuleLaunchKernel.kernelParams = (void**)kernelParams; \
  cb_data.args.hipHccModuleLaunchKernel.extra = (void**)extra; \
  cb_data.args.hipHccModuleLaunchKernel.startEvent = (hipEvent_t)startEvent; \
  cb_data.args.hipHccModuleLaunchKernel.stopEvent = (hipEvent_t)stopEvent; \
};
// hipFree[('void*', 'ptr')]
#define INIT_hipFree_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFree.ptr = (void*)ptr; \
};
// hipOccupancyMaxPotentialBlockSize[('int*', 'gridSize'), ('int*', 'blockSize'), ('const void*', 'f'), ('size_t', 'dynSharedMemPerBlk'), ('int', 'blockSizeLimit')]
#define INIT_hipOccupancyMaxPotentialBlockSize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.gridSize = (int*)gridSize; \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.blockSize = (int*)blockSize; \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.f = (const void*)f; \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.dynSharedMemPerBlk = (size_t)dynSharedMemPerBlk; \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.blockSizeLimit = (int)blockSizeLimit; \
};
// hipDeviceGetAttribute[('int*', 'pi'), ('hipDeviceAttribute_t', 'attr'), ('int', 'deviceId')]
#define INIT_hipDeviceGetAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetAttribute.pi = (int*)pi; \
  cb_data.args.hipDeviceGetAttribute.attr = (hipDeviceAttribute_t)attr; \
  cb_data.args.hipDeviceGetAttribute.deviceId = (int)device; \
};
// hipDeviceComputeCapability[('int*', 'major'), ('int*', 'minor'), ('hipDevice_t', 'device')]
#define INIT_hipDeviceComputeCapability_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceComputeCapability.major = (int*)major; \
  cb_data.args.hipDeviceComputeCapability.minor = (int*)minor; \
  cb_data.args.hipDeviceComputeCapability.device = (hipDevice_t)device; \
};
// hipCtxDisablePeerAccess[('hipCtx_t', 'peerCtx')]
#define INIT_hipCtxDisablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxDisablePeerAccess.peerCtx = (hipCtx_t)peerCtx; \
};
// hipMallocManaged[('void**', 'dev_ptr'), ('size_t', 'size'), ('unsigned int', 'flags')]
#define INIT_hipMallocManaged_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocManaged.dev_ptr = (void**)dev_ptr; \
  cb_data.args.hipMallocManaged.size = (size_t)size; \
  cb_data.args.hipMallocManaged.flags = (unsigned int)flags; \
};
// hipDeviceGetByPCIBusId[('int*', 'device'), ('const char*', 'pciBusId')]
#define INIT_hipDeviceGetByPCIBusId_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetByPCIBusId.device = (int*)device; \
  cb_data.args.hipDeviceGetByPCIBusId.pciBusId = (pciBusIdstr) ? strdup(pciBusIdstr) : NULL; \
};
// hipIpcGetMemHandle[('hipIpcMemHandle_t*', 'handle'), ('void*', 'devPtr')]
#define INIT_hipIpcGetMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcGetMemHandle.handle = (hipIpcMemHandle_t*)handle; \
  cb_data.args.hipIpcGetMemHandle.devPtr = (void*)dev_ptr; \
};
// hipMemcpyHtoDAsync[('hipDeviceptr_t', 'dst'), ('void*', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyHtoDAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoDAsync.dst = (hipDeviceptr_t)dstDevice; \
  cb_data.args.hipMemcpyHtoDAsync.src = (void*)srcHost; \
  cb_data.args.hipMemcpyHtoDAsync.sizeBytes = (size_t)ByteCount; \
  cb_data.args.hipMemcpyHtoDAsync.stream = (hipStream_t)stream; \
};
// hipCtxGetDevice[('hipDevice_t*', 'device')]
#define INIT_hipCtxGetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetDevice.device = (hipDevice_t*)device; \
};
// hipMemcpyDtoD[('hipDeviceptr_t', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes')]
#define INIT_hipMemcpyDtoD_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoD.dst = (hipDeviceptr_t)dstDevice; \
  cb_data.args.hipMemcpyDtoD.src = (hipDeviceptr_t)srcDevice; \
  cb_data.args.hipMemcpyDtoD.sizeBytes = (size_t)ByteCount; \
};
// hipModuleLoadData[('hipModule_t*', 'module'), ('const void*', 'image')]
#define INIT_hipModuleLoadData_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoadData.module = (hipModule_t*)module; \
  cb_data.args.hipModuleLoadData.image = (const void*)image; \
};
// hipDevicePrimaryCtxRelease[('hipDevice_t', 'dev')]
#define INIT_hipDevicePrimaryCtxRelease_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxRelease.dev = (hipDevice_t)dev; \
};
// hipOccupancyMaxActiveBlocksPerMultiprocessor[('int*', 'numBlocks'), ('const void*', 'f'), ('int', 'blockSize'), ('size_t', 'dynamicSMemSize')]
#define INIT_hipOccupancyMaxActiveBlocksPerMultiprocessor_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks = (int*)numBlocks; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessor.f = (const void*)f; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessor.blockSize = (int)blockSize; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessor.dynamicSMemSize = (size_t)dynamicSMemSize; \
};
// hipCtxSetCurrent[('hipCtx_t', 'ctx')]
#define INIT_hipCtxSetCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetCurrent.ctx = (hipCtx_t)ctx; \
};
// hipGetErrorString[]
#define INIT_hipGetErrorString_CB_ARGS_DATA(cb_data) { \
};
// hipStreamCreate[('hipStream_t*', 'stream')]
#define INIT_hipStreamCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreate.stream = (hipStream_t*)stream; \
};
// hipDevicePrimaryCtxRetain[('hipCtx_t*', 'pctx'), ('hipDevice_t', 'dev')]
#define INIT_hipDevicePrimaryCtxRetain_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxRetain.pctx = (hipCtx_t*)pctx; \
  cb_data.args.hipDevicePrimaryCtxRetain.dev = (hipDevice_t)dev; \
};
// hipDeviceGet[('hipDevice_t*', 'device'), ('int', 'ordinal')]
#define INIT_hipDeviceGet_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGet.device = (hipDevice_t*)device; \
  cb_data.args.hipDeviceGet.ordinal = (int)deviceId; \
};
// hipStreamCreateWithFlags[('hipStream_t*', 'stream'), ('unsigned int', 'flags')]
#define INIT_hipStreamCreateWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreateWithFlags.stream = (hipStream_t*)stream; \
  cb_data.args.hipStreamCreateWithFlags.flags = (unsigned int)flags; \
};
// hipMemcpyFromArray[('void*', 'dst'), ('hipArray_const_t', 'srcArray'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'count'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpyFromArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyFromArray.dst = (void*)dst; \
  cb_data.args.hipMemcpyFromArray.srcArray = (hipArray_const_t)src; \
  cb_data.args.hipMemcpyFromArray.wOffset = (size_t)wOffsetSrc; \
  cb_data.args.hipMemcpyFromArray.hOffset = (size_t)hOffset; \
  cb_data.args.hipMemcpyFromArray.count = (size_t)count; \
  cb_data.args.hipMemcpyFromArray.kind = (hipMemcpyKind)kind; \
};
// hipMemcpy2DAsync[('void*', 'dst'), ('size_t', 'dpitch'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpy2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpy2DAsync.dpitch = (size_t)dpitch; \
  cb_data.args.hipMemcpy2DAsync.src = (const void*)src; \
  cb_data.args.hipMemcpy2DAsync.spitch = (size_t)spitch; \
  cb_data.args.hipMemcpy2DAsync.width = (size_t)width; \
  cb_data.args.hipMemcpy2DAsync.height = (size_t)height; \
  cb_data.args.hipMemcpy2DAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpy2DAsync.stream = (hipStream_t)stream; \
};
// hipFuncGetAttributes[('hipFuncAttributes*', 'attr'), ('const void*', 'func')]
#define INIT_hipFuncGetAttributes_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncGetAttributes.attr = (hipFuncAttributes*)attr; \
  cb_data.args.hipFuncGetAttributes.func = (const void*)func; \
};
// hipGetSymbolSize[('size_t*', 'size'), ('const void*', 'symbol')]
#define INIT_hipGetSymbolSize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetSymbolSize.size = (size_t*)sizePtr; \
  cb_data.args.hipGetSymbolSize.symbol = (const void*)symbol; \
};
// hipHostFree[('void*', 'ptr')]
#define INIT_hipHostFree_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostFree.ptr = (void*)ptr; \
};
// hipEventCreateWithFlags[('hipEvent_t*', 'event'), ('unsigned int', 'flags')]
#define INIT_hipEventCreateWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventCreateWithFlags.event = (hipEvent_t*)event; \
  cb_data.args.hipEventCreateWithFlags.flags = (unsigned int)flags; \
};
// hipStreamQuery[('hipStream_t', 'stream')]
#define INIT_hipStreamQuery_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamQuery.stream = (hipStream_t)stream; \
};
// hipMemcpy3D[('const hipMemcpy3DParms*', 'p')]
#define INIT_hipMemcpy3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy3D.p = (const hipMemcpy3DParms*)p; \
};
// hipMemcpyToSymbol[('const void*', 'symbol'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpyToSymbol_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyToSymbol.symbol = (const void*)symbol; \
  cb_data.args.hipMemcpyToSymbol.src = (const void*)src; \
  cb_data.args.hipMemcpyToSymbol.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyToSymbol.offset = (size_t)offset; \
  cb_data.args.hipMemcpyToSymbol.kind = (hipMemcpyKind)kind; \
};
// hipMemcpy[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy.dst = (void*)dst; \
  cb_data.args.hipMemcpy.src = (const void*)src; \
  cb_data.args.hipMemcpy.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpy.kind = (hipMemcpyKind)kind; \
};
// hipPeekAtLastError[]
#define INIT_hipPeekAtLastError_CB_ARGS_DATA(cb_data) { \
};
// hipExtLaunchMultiKernelMultiDevice[('hipLaunchParams*', 'launchParamsList'), ('int', 'numDevices'), ('unsigned int', 'flags')]
#define INIT_hipExtLaunchMultiKernelMultiDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtLaunchMultiKernelMultiDevice.launchParamsList = (hipLaunchParams*)launchParamsList; \
  cb_data.args.hipExtLaunchMultiKernelMultiDevice.numDevices = (int)numDevices; \
  cb_data.args.hipExtLaunchMultiKernelMultiDevice.flags = (unsigned int)flags; \
};
// hipHostAlloc[('void**', 'ptr'), ('size_t', 'size'), ('unsigned int', 'flags')]
#define INIT_hipHostAlloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostAlloc.ptr = (void**)ptr; \
  cb_data.args.hipHostAlloc.size = (size_t)sizeBytes; \
  cb_data.args.hipHostAlloc.flags = (unsigned int)flags; \
};
// hipStreamAddCallback[('hipStream_t', 'stream'), ('hipStreamCallback_t', 'callback'), ('void*', 'userData'), ('unsigned int', 'flags')]
#define INIT_hipStreamAddCallback_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamAddCallback.stream = (hipStream_t)stream; \
  cb_data.args.hipStreamAddCallback.callback = (hipStreamCallback_t)callback; \
  cb_data.args.hipStreamAddCallback.userData = (void*)userData; \
  cb_data.args.hipStreamAddCallback.flags = (unsigned int)flags; \
};
// hipMemcpyToArray[('hipArray*', 'dst'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('const void*', 'src'), ('size_t', 'count'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpyToArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyToArray.dst = (hipArray*)dst; \
  cb_data.args.hipMemcpyToArray.wOffset = (size_t)wOffset; \
  cb_data.args.hipMemcpyToArray.hOffset = (size_t)hOffset; \
  cb_data.args.hipMemcpyToArray.src = (const void*)src; \
  cb_data.args.hipMemcpyToArray.count = (size_t)count; \
  cb_data.args.hipMemcpyToArray.kind = (hipMemcpyKind)kind; \
};
// hipMemsetD32[('hipDeviceptr_t', 'dest'), ('int', 'value'), ('size_t', 'count')]
#define INIT_hipMemsetD32_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD32.dest = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemsetD32.value = (int)value; \
  cb_data.args.hipMemsetD32.count = (size_t)count; \
};
// hipExtModuleLaunchKernel[('hipFunction_t', 'f'), ('unsigned int', 'globalWorkSizeX'), ('unsigned int', 'globalWorkSizeY'), ('unsigned int', 'globalWorkSizeZ'), ('unsigned int', 'localWorkSizeX'), ('unsigned int', 'localWorkSizeY'), ('unsigned int', 'localWorkSizeZ'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'hStream'), ('void**', 'kernelParams'), ('void**', 'extra'), ('hipEvent_t', 'startEvent'), ('hipEvent_t', 'stopEvent'), ('unsigned int', 'flags')]
#define INIT_hipExtModuleLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtModuleLaunchKernel.f = (hipFunction_t)f; \
  cb_data.args.hipExtModuleLaunchKernel.globalWorkSizeX = (unsigned int)globalWorkSizeX; \
  cb_data.args.hipExtModuleLaunchKernel.globalWorkSizeY = (unsigned int)globalWorkSizeY; \
  cb_data.args.hipExtModuleLaunchKernel.globalWorkSizeZ = (unsigned int)globalWorkSizeZ; \
  cb_data.args.hipExtModuleLaunchKernel.localWorkSizeX = (unsigned int)localWorkSizeX; \
  cb_data.args.hipExtModuleLaunchKernel.localWorkSizeY = (unsigned int)localWorkSizeY; \
  cb_data.args.hipExtModuleLaunchKernel.localWorkSizeZ = (unsigned int)localWorkSizeZ; \
  cb_data.args.hipExtModuleLaunchKernel.sharedMemBytes = (size_t)sharedMemBytes; \
  cb_data.args.hipExtModuleLaunchKernel.hStream = (hipStream_t)hStream; \
  cb_data.args.hipExtModuleLaunchKernel.kernelParams = (void**)kernelParams; \
  cb_data.args.hipExtModuleLaunchKernel.extra = (void**)extra; \
  cb_data.args.hipExtModuleLaunchKernel.startEvent = (hipEvent_t)startEvent; \
  cb_data.args.hipExtModuleLaunchKernel.stopEvent = (hipEvent_t)stopEvent; \
  cb_data.args.hipExtModuleLaunchKernel.flags = (unsigned int)flags; \
};
// hipDeviceSynchronize[]
#define INIT_hipDeviceSynchronize_CB_ARGS_DATA(cb_data) { \
};
// hipDeviceGetCacheConfig[('hipFuncCache_t*', 'cacheConfig')]
#define INIT_hipDeviceGetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetCacheConfig.cacheConfig = (hipFuncCache_t*)cacheConfig; \
};
// hipMalloc3D[('hipPitchedPtr*', 'pitchedDevPtr'), ('hipExtent', 'extent')]
#define INIT_hipMalloc3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc3D.pitchedDevPtr = (hipPitchedPtr*)pitchedDevPtr; \
  cb_data.args.hipMalloc3D.extent = (hipExtent)extent; \
};
// hipPointerGetAttributes[('hipPointerAttribute_t*', 'attributes'), ('const void*', 'ptr')]
#define INIT_hipPointerGetAttributes_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipPointerGetAttributes.attributes = (hipPointerAttribute_t*)attributes; \
  cb_data.args.hipPointerGetAttributes.ptr = (const void*)ptr; \
};
// hipMemsetAsync[('void*', 'dst'), ('int', 'value'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemsetAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetAsync.dst = (void*)dst; \
  cb_data.args.hipMemsetAsync.value = (int)value; \
  cb_data.args.hipMemsetAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemsetAsync.stream = (hipStream_t)stream; \
};
// hipDeviceGetName[('char*', 'name'), ('int', 'len'), ('hipDevice_t', 'device')]
#define INIT_hipDeviceGetName_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetName.name = (char*)name; \
  cb_data.args.hipDeviceGetName.len = (int)len; \
  cb_data.args.hipDeviceGetName.device = (hipDevice_t)device; \
};
// hipModuleOccupancyMaxPotentialBlockSizeWithFlags[('int*', 'gridSize'), ('int*', 'blockSize'), ('hipFunction_t', 'f'), ('size_t', 'dynSharedMemPerBlk'), ('int', 'blockSizeLimit'), ('unsigned int', 'flags')]
#define INIT_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize = (int*)gridSize; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize = (int*)blockSize; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.f = (hipFunction_t)f; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.dynSharedMemPerBlk = (size_t)dynSharedMemPerBlk; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSizeLimit = (int)blockSizeLimit; \
  cb_data.args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.flags = (unsigned int)flags; \
};
// hipCtxPushCurrent[('hipCtx_t', 'ctx')]
#define INIT_hipCtxPushCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxPushCurrent.ctx = (hipCtx_t)ctx; \
};
// hipMemcpyPeer[('void*', 'dst'), ('int', 'dstDeviceId'), ('const void*', 'src'), ('int', 'srcDeviceId'), ('size_t', 'sizeBytes')]
#define INIT_hipMemcpyPeer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyPeer.dst = (void*)dst; \
  cb_data.args.hipMemcpyPeer.dstDeviceId = (int)dstDevice; \
  cb_data.args.hipMemcpyPeer.src = (const void*)src; \
  cb_data.args.hipMemcpyPeer.srcDeviceId = (int)srcDevice; \
  cb_data.args.hipMemcpyPeer.sizeBytes = (size_t)sizeBytes; \
};
// hipEventSynchronize[('hipEvent_t', 'event')]
#define INIT_hipEventSynchronize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventSynchronize.event = (hipEvent_t)event; \
};
// hipMemcpyDtoDAsync[('hipDeviceptr_t', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyDtoDAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoDAsync.dst = (hipDeviceptr_t)dstDevice; \
  cb_data.args.hipMemcpyDtoDAsync.src = (hipDeviceptr_t)srcDevice; \
  cb_data.args.hipMemcpyDtoDAsync.sizeBytes = (size_t)ByteCount; \
  cb_data.args.hipMemcpyDtoDAsync.stream = (hipStream_t)stream; \
};
// hipProfilerStart[]
#define INIT_hipProfilerStart_CB_ARGS_DATA(cb_data) { \
};
// hipExtMallocWithFlags[('void**', 'ptr'), ('size_t', 'sizeBytes'), ('unsigned int', 'flags')]
#define INIT_hipExtMallocWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtMallocWithFlags.ptr = (void**)ptr; \
  cb_data.args.hipExtMallocWithFlags.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipExtMallocWithFlags.flags = (unsigned int)flags; \
};
// hipCtxEnablePeerAccess[('hipCtx_t', 'peerCtx'), ('unsigned int', 'flags')]
#define INIT_hipCtxEnablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxEnablePeerAccess.peerCtx = (hipCtx_t)peerCtx; \
  cb_data.args.hipCtxEnablePeerAccess.flags = (unsigned int)flags; \
};
// hipMemAllocHost[('void**', 'ptr'), ('size_t', 'size')]
#define INIT_hipMemAllocHost_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemAllocHost.ptr = (void**)ptr; \
  cb_data.args.hipMemAllocHost.size = (size_t)size; \
};
// hipMemcpyDtoHAsync[('void*', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyDtoHAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoHAsync.dst = (void*)dstHost; \
  cb_data.args.hipMemcpyDtoHAsync.src = (hipDeviceptr_t)srcDevice; \
  cb_data.args.hipMemcpyDtoHAsync.sizeBytes = (size_t)ByteCount; \
  cb_data.args.hipMemcpyDtoHAsync.stream = (hipStream_t)stream; \
};
// hipModuleLaunchKernel[('hipFunction_t', 'f'), ('unsigned int', 'gridDimX'), ('unsigned int', 'gridDimY'), ('unsigned int', 'gridDimZ'), ('unsigned int', 'blockDimX'), ('unsigned int', 'blockDimY'), ('unsigned int', 'blockDimZ'), ('unsigned int', 'sharedMemBytes'), ('hipStream_t', 'stream'), ('void**', 'kernelParams'), ('void**', 'extra')]
#define INIT_hipModuleLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLaunchKernel.f = (hipFunction_t)f; \
  cb_data.args.hipModuleLaunchKernel.gridDimX = (unsigned int)gridDimX; \
  cb_data.args.hipModuleLaunchKernel.gridDimY = (unsigned int)gridDimY; \
  cb_data.args.hipModuleLaunchKernel.gridDimZ = (unsigned int)gridDimZ; \
  cb_data.args.hipModuleLaunchKernel.blockDimX = (unsigned int)blockDimX; \
  cb_data.args.hipModuleLaunchKernel.blockDimY = (unsigned int)blockDimY; \
  cb_data.args.hipModuleLaunchKernel.blockDimZ = (unsigned int)blockDimZ; \
  cb_data.args.hipModuleLaunchKernel.sharedMemBytes = (unsigned int)sharedMemBytes; \
  cb_data.args.hipModuleLaunchKernel.stream = (hipStream_t)hStream; \
  cb_data.args.hipModuleLaunchKernel.kernelParams = (void**)kernelParams; \
  cb_data.args.hipModuleLaunchKernel.extra = (void**)extra; \
};
// hipMemAllocPitch[('hipDeviceptr_t*', 'dptr'), ('size_t*', 'pitch'), ('size_t', 'widthInBytes'), ('size_t', 'height'), ('unsigned int', 'elementSizeBytes')]
#define INIT_hipMemAllocPitch_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemAllocPitch.dptr = (hipDeviceptr_t*)dptr; \
  cb_data.args.hipMemAllocPitch.pitch = (size_t*)pitch; \
  cb_data.args.hipMemAllocPitch.widthInBytes = (size_t)widthInBytes; \
  cb_data.args.hipMemAllocPitch.height = (size_t)height; \
  cb_data.args.hipMemAllocPitch.elementSizeBytes = (unsigned int)elementSizeBytes; \
};
// hipExtLaunchKernel[('const void*', 'function_address'), ('dim3', 'numBlocks'), ('dim3', 'dimBlocks'), ('void**', 'args'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'stream'), ('hipEvent_t', 'startEvent'), ('hipEvent_t', 'stopEvent'), ('int', 'flags')]
#define INIT_hipExtLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtLaunchKernel.function_address = (const void*)hostFunction; \
  cb_data.args.hipExtLaunchKernel.numBlocks = (dim3)gridDim; \
  cb_data.args.hipExtLaunchKernel.dimBlocks = (dim3)blockDim; \
  cb_data.args.hipExtLaunchKernel.args = (void**)args; \
  cb_data.args.hipExtLaunchKernel.sharedMemBytes = (size_t)sharedMemBytes; \
  cb_data.args.hipExtLaunchKernel.stream = (hipStream_t)stream; \
  cb_data.args.hipExtLaunchKernel.startEvent = (hipEvent_t)startEvent; \
  cb_data.args.hipExtLaunchKernel.stopEvent = (hipEvent_t)stopEvent; \
  cb_data.args.hipExtLaunchKernel.flags = (int)flags; \
};
// hipMemcpy2DFromArrayAsync[('void*', 'dst'), ('size_t', 'dpitch'), ('hipArray_const_t', 'src'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpy2DFromArrayAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DFromArrayAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpy2DFromArrayAsync.dpitch = (size_t)dpitch; \
  cb_data.args.hipMemcpy2DFromArrayAsync.src = (hipArray_const_t)src; \
  cb_data.args.hipMemcpy2DFromArrayAsync.wOffset = (size_t)wOffsetSrc; \
  cb_data.args.hipMemcpy2DFromArrayAsync.hOffset = (size_t)hOffsetSrc; \
  cb_data.args.hipMemcpy2DFromArrayAsync.width = (size_t)width; \
  cb_data.args.hipMemcpy2DFromArrayAsync.height = (size_t)height; \
  cb_data.args.hipMemcpy2DFromArrayAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpy2DFromArrayAsync.stream = (hipStream_t)stream; \
};
// hipDeviceGetLimit[('size_t*', 'pValue'), ('hipLimit_t', 'limit')]
#define INIT_hipDeviceGetLimit_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetLimit.pValue = (size_t*)pValue; \
  cb_data.args.hipDeviceGetLimit.limit = (hipLimit_t)limit; \
};
// hipModuleLoadDataEx[('hipModule_t*', 'module'), ('const void*', 'image'), ('unsigned int', 'numOptions'), ('hipJitOption*', 'options'), ('void**', 'optionsValues')]
#define INIT_hipModuleLoadDataEx_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoadDataEx.module = (hipModule_t*)module; \
  cb_data.args.hipModuleLoadDataEx.image = (const void*)image; \
  cb_data.args.hipModuleLoadDataEx.numOptions = (unsigned int)numOptions; \
  cb_data.args.hipModuleLoadDataEx.options = (hipJitOption*)options; \
  cb_data.args.hipModuleLoadDataEx.optionsValues = (void**)optionsValues; \
};
// hipRuntimeGetVersion[('int*', 'runtimeVersion')]
#define INIT_hipRuntimeGetVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipRuntimeGetVersion.runtimeVersion = (int*)runtimeVersion; \
};
// hipMemRangeGetAttribute[('void*', 'data'), ('size_t', 'data_size'), ('hipMemRangeAttribute', 'attribute'), ('const void*', 'dev_ptr'), ('size_t', 'count')]
#define INIT_hipMemRangeGetAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemRangeGetAttribute.data = (void*)data; \
  cb_data.args.hipMemRangeGetAttribute.data_size = (size_t)data_size; \
  cb_data.args.hipMemRangeGetAttribute.attribute = (hipMemRangeAttribute)attribute; \
  cb_data.args.hipMemRangeGetAttribute.dev_ptr = (const void*)dev_ptr; \
  cb_data.args.hipMemRangeGetAttribute.count = (size_t)count; \
};
// hipDeviceGetP2PAttribute[('int*', 'value'), ('hipDeviceP2PAttr', 'attr'), ('int', 'srcDevice'), ('int', 'dstDevice')]
#define INIT_hipDeviceGetP2PAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetP2PAttribute.value = (int*)value; \
  cb_data.args.hipDeviceGetP2PAttribute.attr = (hipDeviceP2PAttr)attr; \
  cb_data.args.hipDeviceGetP2PAttribute.srcDevice = (int)srcDevice; \
  cb_data.args.hipDeviceGetP2PAttribute.dstDevice = (int)dstDevice; \
};
// hipMemcpyPeerAsync[('void*', 'dst'), ('int', 'dstDeviceId'), ('const void*', 'src'), ('int', 'srcDevice'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyPeerAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyPeerAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpyPeerAsync.dstDeviceId = (int)dstDevice; \
  cb_data.args.hipMemcpyPeerAsync.src = (const void*)src; \
  cb_data.args.hipMemcpyPeerAsync.srcDevice = (int)srcDevice; \
  cb_data.args.hipMemcpyPeerAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyPeerAsync.stream = (hipStream_t)stream; \
};
// hipGetDeviceProperties[('hipDeviceProp_t*', 'props'), ('hipDevice_t', 'device')]
#define INIT_hipGetDeviceProperties_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceProperties.props = (hipDeviceProp_t*)props; \
  cb_data.args.hipGetDeviceProperties.device = (hipDevice_t)device; \
};
// hipMemcpyDtoH[('void*', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes')]
#define INIT_hipMemcpyDtoH_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoH.dst = (void*)dstHost; \
  cb_data.args.hipMemcpyDtoH.src = (hipDeviceptr_t)srcDevice; \
  cb_data.args.hipMemcpyDtoH.sizeBytes = (size_t)ByteCount; \
};
// hipMemcpyWithStream[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyWithStream_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyWithStream.dst = (void*)dst; \
  cb_data.args.hipMemcpyWithStream.src = (const void*)src; \
  cb_data.args.hipMemcpyWithStream.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyWithStream.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpyWithStream.stream = (hipStream_t)stream; \
};
// hipDeviceTotalMem[('size_t*', 'bytes'), ('hipDevice_t', 'device')]
#define INIT_hipDeviceTotalMem_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceTotalMem.bytes = (size_t*)bytes; \
  cb_data.args.hipDeviceTotalMem.device = (hipDevice_t)device; \
};
// hipHostGetDevicePointer[('void**', 'devPtr'), ('void*', 'hstPtr'), ('unsigned int', 'flags')]
#define INIT_hipHostGetDevicePointer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostGetDevicePointer.devPtr = (void**)devicePointer; \
  cb_data.args.hipHostGetDevicePointer.hstPtr = (void*)hostPointer; \
  cb_data.args.hipHostGetDevicePointer.flags = (unsigned int)flags; \
};
// hipMemRangeGetAttributes[('void**', 'data'), ('size_t*', 'data_sizes'), ('hipMemRangeAttribute*', 'attributes'), ('size_t', 'num_attributes'), ('const void*', 'dev_ptr'), ('size_t', 'count')]
#define INIT_hipMemRangeGetAttributes_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemRangeGetAttributes.data = (void**)data; \
  cb_data.args.hipMemRangeGetAttributes.data_sizes = (size_t*)data_sizes; \
  cb_data.args.hipMemRangeGetAttributes.attributes = (hipMemRangeAttribute*)attributes; \
  cb_data.args.hipMemRangeGetAttributes.num_attributes = (size_t)num_attributes; \
  cb_data.args.hipMemRangeGetAttributes.dev_ptr = (const void*)dev_ptr; \
  cb_data.args.hipMemRangeGetAttributes.count = (size_t)count; \
};
// hipMemcpyParam2D[('const hip_Memcpy2D*', 'pCopy')]
#define INIT_hipMemcpyParam2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyParam2D.pCopy = (const hip_Memcpy2D*)pCopy; \
};
// hipDevicePrimaryCtxReset[('hipDevice_t', 'dev')]
#define INIT_hipDevicePrimaryCtxReset_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxReset.dev = (hipDevice_t)dev; \
};
// hipGetMipmappedArrayLevel[('hipArray_t*', 'levelArray'), ('hipMipmappedArray_const_t', 'mipmappedArray'), ('unsigned int', 'level')]
#define INIT_hipGetMipmappedArrayLevel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetMipmappedArrayLevel.levelArray = (hipArray_t*)levelArray; \
  cb_data.args.hipGetMipmappedArrayLevel.mipmappedArray = (hipMipmappedArray_const_t)mipmappedArray; \
  cb_data.args.hipGetMipmappedArrayLevel.level = (unsigned int)level; \
};
// hipMemsetD32Async[('hipDeviceptr_t', 'dst'), ('int', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
#define INIT_hipMemsetD32Async_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD32Async.dst = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemsetD32Async.value = (int)value; \
  cb_data.args.hipMemsetD32Async.count = (size_t)count; \
  cb_data.args.hipMemsetD32Async.stream = (hipStream_t)stream; \
};
// hipGetDevice[('int*', 'deviceId')]
#define INIT_hipGetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDevice.deviceId = (int*)deviceId; \
};
// hipGetDeviceCount[('int*', 'count')]
#define INIT_hipGetDeviceCount_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceCount.count = (int*)count; \
};
// hipIpcOpenEventHandle[('hipEvent_t*', 'event'), ('hipIpcEventHandle_t', 'handle')]
#define INIT_hipIpcOpenEventHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcOpenEventHandle.event = (hipEvent_t*)event; \
  cb_data.args.hipIpcOpenEventHandle.handle = (hipIpcEventHandle_t)handle; \
};
#define INIT_CB_ARGS_DATA(cb_id, cb_data) INIT_##cb_id##_CB_ARGS_DATA(cb_data)
#if HIP_PROF_HIP_API_STRING

// HIP API args filling method
static inline void hipApiArgsInit(hip_api_id_t id, hip_api_data_t* data) {
  switch (id) {
// hipDrvMemcpy3DAsync[('const HIP_MEMCPY3D*', 'pCopy'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipDrvMemcpy3DAsync:
      if (data->args.hipDrvMemcpy3DAsync.pCopy) data->args.hipDrvMemcpy3DAsync.pCopy__val = *(data->args.hipDrvMemcpy3DAsync.pCopy);
      break;
// hipDeviceEnablePeerAccess[('int', 'peerDeviceId'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipDeviceEnablePeerAccess:
      break;
// hipFuncSetSharedMemConfig[('const void*', 'func'), ('hipSharedMemConfig', 'config')]
    case HIP_API_ID_hipFuncSetSharedMemConfig:
      break;
// hipMemcpyToSymbolAsync[('const void*', 'symbol'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyToSymbolAsync:
      break;
// hipMallocPitch[('void**', 'ptr'), ('size_t*', 'pitch'), ('size_t', 'width'), ('size_t', 'height')]
    case HIP_API_ID_hipMallocPitch:
      if (data->args.hipMallocPitch.ptr) data->args.hipMallocPitch.ptr__val = *(data->args.hipMallocPitch.ptr);
      if (data->args.hipMallocPitch.pitch) data->args.hipMallocPitch.pitch__val = *(data->args.hipMallocPitch.pitch);
      break;
// hipMalloc[('void**', 'ptr'), ('size_t', 'size')]
    case HIP_API_ID_hipMalloc:
      if (data->args.hipMalloc.ptr) data->args.hipMalloc.ptr__val = *(data->args.hipMalloc.ptr);
      break;
// hipMemsetD16[('hipDeviceptr_t', 'dest'), ('unsigned short', 'value'), ('size_t', 'count')]
    case HIP_API_ID_hipMemsetD16:
      break;
// hipExtStreamGetCUMask[('hipStream_t', 'stream'), ('unsigned int', 'cuMaskSize'), ('unsigned int*', 'cuMask')]
    case HIP_API_ID_hipExtStreamGetCUMask:
      if (data->args.hipExtStreamGetCUMask.cuMask) data->args.hipExtStreamGetCUMask.cuMask__val = *(data->args.hipExtStreamGetCUMask.cuMask);
      break;
// hipEventRecord[('hipEvent_t', 'event'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipEventRecord:
      break;
// hipCtxSynchronize[]
    case HIP_API_ID_hipCtxSynchronize:
      break;
// hipSetDevice[('int', 'deviceId')]
    case HIP_API_ID_hipSetDevice:
      break;
// hipCtxGetApiVersion[('hipCtx_t', 'ctx'), ('int*', 'apiVersion')]
    case HIP_API_ID_hipCtxGetApiVersion:
      if (data->args.hipCtxGetApiVersion.apiVersion) data->args.hipCtxGetApiVersion.apiVersion__val = *(data->args.hipCtxGetApiVersion.apiVersion);
      break;
// hipMemcpyFromSymbolAsync[('void*', 'dst'), ('const void*', 'symbol'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyFromSymbolAsync:
      break;
// hipExtGetLinkTypeAndHopCount[('int', 'device1'), ('int', 'device2'), ('unsigned int*', 'linktype'), ('unsigned int*', 'hopcount')]
    case HIP_API_ID_hipExtGetLinkTypeAndHopCount:
      if (data->args.hipExtGetLinkTypeAndHopCount.linktype) data->args.hipExtGetLinkTypeAndHopCount.linktype__val = *(data->args.hipExtGetLinkTypeAndHopCount.linktype);
      if (data->args.hipExtGetLinkTypeAndHopCount.hopcount) data->args.hipExtGetLinkTypeAndHopCount.hopcount__val = *(data->args.hipExtGetLinkTypeAndHopCount.hopcount);
      break;
// __hipPopCallConfiguration[('dim3*', 'gridDim'), ('dim3*', 'blockDim'), ('size_t*', 'sharedMem'), ('hipStream_t*', 'stream')]
    case HIP_API_ID___hipPopCallConfiguration:
      if (data->args.__hipPopCallConfiguration.gridDim) data->args.__hipPopCallConfiguration.gridDim__val = *(data->args.__hipPopCallConfiguration.gridDim);
      if (data->args.__hipPopCallConfiguration.blockDim) data->args.__hipPopCallConfiguration.blockDim__val = *(data->args.__hipPopCallConfiguration.blockDim);
      if (data->args.__hipPopCallConfiguration.sharedMem) data->args.__hipPopCallConfiguration.sharedMem__val = *(data->args.__hipPopCallConfiguration.sharedMem);
      if (data->args.__hipPopCallConfiguration.stream) data->args.__hipPopCallConfiguration.stream__val = *(data->args.__hipPopCallConfiguration.stream);
      break;
// hipModuleOccupancyMaxActiveBlocksPerMultiprocessor[('int*', 'numBlocks'), ('hipFunction_t', 'f'), ('int', 'blockSize'), ('size_t', 'dynSharedMemPerBlk')]
    case HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor:
      if (data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks) data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks__val = *(data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks);
      break;
// hipMemset3D[('hipPitchedPtr', 'pitchedDevPtr'), ('int', 'value'), ('hipExtent', 'extent')]
    case HIP_API_ID_hipMemset3D:
      break;
// hipStreamCreateWithPriority[('hipStream_t*', 'stream'), ('unsigned int', 'flags'), ('int', 'priority')]
    case HIP_API_ID_hipStreamCreateWithPriority:
      if (data->args.hipStreamCreateWithPriority.stream) data->args.hipStreamCreateWithPriority.stream__val = *(data->args.hipStreamCreateWithPriority.stream);
      break;
// hipMemcpy2DToArray[('hipArray*', 'dst'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
    case HIP_API_ID_hipMemcpy2DToArray:
      if (data->args.hipMemcpy2DToArray.dst) data->args.hipMemcpy2DToArray.dst__val = *(data->args.hipMemcpy2DToArray.dst);
      break;
// hipMemsetD8Async[('hipDeviceptr_t', 'dest'), ('unsigned char', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemsetD8Async:
      break;
// hipCtxGetCacheConfig[('hipFuncCache_t*', 'cacheConfig')]
    case HIP_API_ID_hipCtxGetCacheConfig:
      if (data->args.hipCtxGetCacheConfig.cacheConfig) data->args.hipCtxGetCacheConfig.cacheConfig__val = *(data->args.hipCtxGetCacheConfig.cacheConfig);
      break;
// hipModuleGetFunction[('hipFunction_t*', 'function'), ('hipModule_t', 'module'), ('const char*', 'kname')]
    case HIP_API_ID_hipModuleGetFunction:
      if (data->args.hipModuleGetFunction.function) data->args.hipModuleGetFunction.function__val = *(data->args.hipModuleGetFunction.function);
      if (data->args.hipModuleGetFunction.kname) data->args.hipModuleGetFunction.kname__val = *(data->args.hipModuleGetFunction.kname);
      break;
// hipStreamWaitEvent[('hipStream_t', 'stream'), ('hipEvent_t', 'event'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipStreamWaitEvent:
      break;
// hipDeviceGetStreamPriorityRange[('int*', 'leastPriority'), ('int*', 'greatestPriority')]
    case HIP_API_ID_hipDeviceGetStreamPriorityRange:
      if (data->args.hipDeviceGetStreamPriorityRange.leastPriority) data->args.hipDeviceGetStreamPriorityRange.leastPriority__val = *(data->args.hipDeviceGetStreamPriorityRange.leastPriority);
      if (data->args.hipDeviceGetStreamPriorityRange.greatestPriority) data->args.hipDeviceGetStreamPriorityRange.greatestPriority__val = *(data->args.hipDeviceGetStreamPriorityRange.greatestPriority);
      break;
// hipModuleLoad[('hipModule_t*', 'module'), ('const char*', 'fname')]
    case HIP_API_ID_hipModuleLoad:
      if (data->args.hipModuleLoad.module) data->args.hipModuleLoad.module__val = *(data->args.hipModuleLoad.module);
      if (data->args.hipModuleLoad.fname) data->args.hipModuleLoad.fname__val = *(data->args.hipModuleLoad.fname);
      break;
// hipDevicePrimaryCtxSetFlags[('hipDevice_t', 'dev'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipDevicePrimaryCtxSetFlags:
      break;
// hipLaunchCooperativeKernel[('const void*', 'f'), ('dim3', 'gridDim'), ('dim3', 'blockDimX'), ('void**', 'kernelParams'), ('unsigned int', 'sharedMemBytes'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipLaunchCooperativeKernel:
      if (data->args.hipLaunchCooperativeKernel.kernelParams) data->args.hipLaunchCooperativeKernel.kernelParams__val = *(data->args.hipLaunchCooperativeKernel.kernelParams);
      break;
// hipLaunchCooperativeKernelMultiDevice[('hipLaunchParams*', 'launchParamsList'), ('int', 'numDevices'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
      if (data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList) data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val = *(data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList);
      break;
// hipMemcpyAsync[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyAsync:
      break;
// hipMalloc3DArray[('hipArray_t*', 'array'), ('const hipChannelFormatDesc*', 'desc'), ('hipExtent', 'extent'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipMalloc3DArray:
      if (data->args.hipMalloc3DArray.array) data->args.hipMalloc3DArray.array__val = *(data->args.hipMalloc3DArray.array);
      if (data->args.hipMalloc3DArray.desc) data->args.hipMalloc3DArray.desc__val = *(data->args.hipMalloc3DArray.desc);
      break;
// hipMallocHost[('void**', 'ptr'), ('size_t', 'size')]
    case HIP_API_ID_hipMallocHost:
      if (data->args.hipMallocHost.ptr) data->args.hipMallocHost.ptr__val = *(data->args.hipMallocHost.ptr);
      break;
// hipCtxGetCurrent[('hipCtx_t*', 'ctx')]
    case HIP_API_ID_hipCtxGetCurrent:
      if (data->args.hipCtxGetCurrent.ctx) data->args.hipCtxGetCurrent.ctx__val = *(data->args.hipCtxGetCurrent.ctx);
      break;
// hipDevicePrimaryCtxGetState[('hipDevice_t', 'dev'), ('unsigned int*', 'flags'), ('int*', 'active')]
    case HIP_API_ID_hipDevicePrimaryCtxGetState:
      if (data->args.hipDevicePrimaryCtxGetState.flags) data->args.hipDevicePrimaryCtxGetState.flags__val = *(data->args.hipDevicePrimaryCtxGetState.flags);
      if (data->args.hipDevicePrimaryCtxGetState.active) data->args.hipDevicePrimaryCtxGetState.active__val = *(data->args.hipDevicePrimaryCtxGetState.active);
      break;
// hipEventQuery[('hipEvent_t', 'event')]
    case HIP_API_ID_hipEventQuery:
      break;
// hipEventCreate[('hipEvent_t*', 'event')]
    case HIP_API_ID_hipEventCreate:
      if (data->args.hipEventCreate.event) data->args.hipEventCreate.event__val = *(data->args.hipEventCreate.event);
      break;
// hipMemGetAddressRange[('hipDeviceptr_t*', 'pbase'), ('size_t*', 'psize'), ('hipDeviceptr_t', 'dptr')]
    case HIP_API_ID_hipMemGetAddressRange:
      if (data->args.hipMemGetAddressRange.pbase) data->args.hipMemGetAddressRange.pbase__val = *(data->args.hipMemGetAddressRange.pbase);
      if (data->args.hipMemGetAddressRange.psize) data->args.hipMemGetAddressRange.psize__val = *(data->args.hipMemGetAddressRange.psize);
      break;
// hipMemcpyFromSymbol[('void*', 'dst'), ('const void*', 'symbol'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind')]
    case HIP_API_ID_hipMemcpyFromSymbol:
      break;
// hipArrayCreate[('hipArray**', 'pHandle'), ('const HIP_ARRAY_DESCRIPTOR*', 'pAllocateArray')]
    case HIP_API_ID_hipArrayCreate:
      if (data->args.hipArrayCreate.pHandle) data->args.hipArrayCreate.pHandle__val = *(data->args.hipArrayCreate.pHandle);
      if (data->args.hipArrayCreate.pAllocateArray) data->args.hipArrayCreate.pAllocateArray__val = *(data->args.hipArrayCreate.pAllocateArray);
      break;
// hipStreamAttachMemAsync[('hipStream_t', 'stream'), ('hipDeviceptr_t*', 'dev_ptr'), ('size_t', 'length'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipStreamAttachMemAsync:
      if (data->args.hipStreamAttachMemAsync.dev_ptr) data->args.hipStreamAttachMemAsync.dev_ptr__val = *(data->args.hipStreamAttachMemAsync.dev_ptr);
      break;
// hipStreamGetFlags[('hipStream_t', 'stream'), ('unsigned int*', 'flags')]
    case HIP_API_ID_hipStreamGetFlags:
      if (data->args.hipStreamGetFlags.flags) data->args.hipStreamGetFlags.flags__val = *(data->args.hipStreamGetFlags.flags);
      break;
// hipMallocArray[('hipArray**', 'array'), ('const hipChannelFormatDesc*', 'desc'), ('size_t', 'width'), ('size_t', 'height'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipMallocArray:
      if (data->args.hipMallocArray.array) data->args.hipMallocArray.array__val = *(data->args.hipMallocArray.array);
      if (data->args.hipMallocArray.desc) data->args.hipMallocArray.desc__val = *(data->args.hipMallocArray.desc);
      break;
// hipCtxGetSharedMemConfig[('hipSharedMemConfig*', 'pConfig')]
    case HIP_API_ID_hipCtxGetSharedMemConfig:
      if (data->args.hipCtxGetSharedMemConfig.pConfig) data->args.hipCtxGetSharedMemConfig.pConfig__val = *(data->args.hipCtxGetSharedMemConfig.pConfig);
      break;
// hipDeviceDisablePeerAccess[('int', 'peerDeviceId')]
    case HIP_API_ID_hipDeviceDisablePeerAccess:
      break;
// hipModuleOccupancyMaxPotentialBlockSize[('int*', 'gridSize'), ('int*', 'blockSize'), ('hipFunction_t', 'f'), ('size_t', 'dynSharedMemPerBlk'), ('int', 'blockSizeLimit')]
    case HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize:
      if (data->args.hipModuleOccupancyMaxPotentialBlockSize.gridSize) data->args.hipModuleOccupancyMaxPotentialBlockSize.gridSize__val = *(data->args.hipModuleOccupancyMaxPotentialBlockSize.gridSize);
      if (data->args.hipModuleOccupancyMaxPotentialBlockSize.blockSize) data->args.hipModuleOccupancyMaxPotentialBlockSize.blockSize__val = *(data->args.hipModuleOccupancyMaxPotentialBlockSize.blockSize);
      break;
// hipMemPtrGetInfo[('void*', 'ptr'), ('size_t*', 'size')]
    case HIP_API_ID_hipMemPtrGetInfo:
      if (data->args.hipMemPtrGetInfo.size) data->args.hipMemPtrGetInfo.size__val = *(data->args.hipMemPtrGetInfo.size);
      break;
// hipFuncGetAttribute[('int*', 'value'), ('hipFunction_attribute', 'attrib'), ('hipFunction_t', 'hfunc')]
    case HIP_API_ID_hipFuncGetAttribute:
      if (data->args.hipFuncGetAttribute.value) data->args.hipFuncGetAttribute.value__val = *(data->args.hipFuncGetAttribute.value);
      break;
// hipCtxGetFlags[('unsigned int*', 'flags')]
    case HIP_API_ID_hipCtxGetFlags:
      if (data->args.hipCtxGetFlags.flags) data->args.hipCtxGetFlags.flags__val = *(data->args.hipCtxGetFlags.flags);
      break;
// hipStreamDestroy[('hipStream_t', 'stream')]
    case HIP_API_ID_hipStreamDestroy:
      break;
// __hipPushCallConfiguration[('dim3', 'gridDim'), ('dim3', 'blockDim'), ('size_t', 'sharedMem'), ('hipStream_t', 'stream')]
    case HIP_API_ID___hipPushCallConfiguration:
      break;
// hipMemset3DAsync[('hipPitchedPtr', 'pitchedDevPtr'), ('int', 'value'), ('hipExtent', 'extent'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemset3DAsync:
      break;
// hipDeviceGetPCIBusId[('char*', 'pciBusId'), ('int', 'len'), ('int', 'device')]
    case HIP_API_ID_hipDeviceGetPCIBusId:
      data->args.hipDeviceGetPCIBusId.pciBusId = (data->args.hipDeviceGetPCIBusId.pciBusId) ? strdup(data->args.hipDeviceGetPCIBusId.pciBusId) : NULL;
      break;
// hipInit[('unsigned int', 'flags')]
    case HIP_API_ID_hipInit:
      break;
// hipMemcpyAtoH[('void*', 'dst'), ('hipArray*', 'srcArray'), ('size_t', 'srcOffset'), ('size_t', 'count')]
    case HIP_API_ID_hipMemcpyAtoH:
      if (data->args.hipMemcpyAtoH.srcArray) data->args.hipMemcpyAtoH.srcArray__val = *(data->args.hipMemcpyAtoH.srcArray);
      break;
// hipStreamGetPriority[('hipStream_t', 'stream'), ('int*', 'priority')]
    case HIP_API_ID_hipStreamGetPriority:
      if (data->args.hipStreamGetPriority.priority) data->args.hipStreamGetPriority.priority__val = *(data->args.hipStreamGetPriority.priority);
      break;
// hipMemset2D[('void*', 'dst'), ('size_t', 'pitch'), ('int', 'value'), ('size_t', 'width'), ('size_t', 'height')]
    case HIP_API_ID_hipMemset2D:
      break;
// hipMemset2DAsync[('void*', 'dst'), ('size_t', 'pitch'), ('int', 'value'), ('size_t', 'width'), ('size_t', 'height'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemset2DAsync:
      break;
// hipDeviceCanAccessPeer[('int*', 'canAccessPeer'), ('int', 'deviceId'), ('int', 'peerDeviceId')]
    case HIP_API_ID_hipDeviceCanAccessPeer:
      if (data->args.hipDeviceCanAccessPeer.canAccessPeer) data->args.hipDeviceCanAccessPeer.canAccessPeer__val = *(data->args.hipDeviceCanAccessPeer.canAccessPeer);
      break;
// hipLaunchByPtr[('const void*', 'hostFunction')]
    case HIP_API_ID_hipLaunchByPtr:
      break;
// hipMemPrefetchAsync[('const void*', 'dev_ptr'), ('size_t', 'count'), ('int', 'device'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemPrefetchAsync:
      break;
// hipCtxDestroy[('hipCtx_t', 'ctx')]
    case HIP_API_ID_hipCtxDestroy:
      break;
// hipMemsetD16Async[('hipDeviceptr_t', 'dest'), ('unsigned short', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemsetD16Async:
      break;
// hipModuleUnload[('hipModule_t', 'module')]
    case HIP_API_ID_hipModuleUnload:
      break;
// hipHostUnregister[('void*', 'hostPtr')]
    case HIP_API_ID_hipHostUnregister:
      break;
// hipProfilerStop[]
    case HIP_API_ID_hipProfilerStop:
      break;
// hipExtStreamCreateWithCUMask[('hipStream_t*', 'stream'), ('unsigned int', 'cuMaskSize'), ('const unsigned int*', 'cuMask')]
    case HIP_API_ID_hipExtStreamCreateWithCUMask:
      if (data->args.hipExtStreamCreateWithCUMask.stream) data->args.hipExtStreamCreateWithCUMask.stream__val = *(data->args.hipExtStreamCreateWithCUMask.stream);
      if (data->args.hipExtStreamCreateWithCUMask.cuMask) data->args.hipExtStreamCreateWithCUMask.cuMask__val = *(data->args.hipExtStreamCreateWithCUMask.cuMask);
      break;
// hipStreamSynchronize[('hipStream_t', 'stream')]
    case HIP_API_ID_hipStreamSynchronize:
      break;
// hipFreeHost[('void*', 'ptr')]
    case HIP_API_ID_hipFreeHost:
      break;
// hipDeviceSetCacheConfig[('hipFuncCache_t', 'cacheConfig')]
    case HIP_API_ID_hipDeviceSetCacheConfig:
      break;
// hipGetErrorName[]
    case HIP_API_ID_hipGetErrorName:
      break;
// hipMemcpyHtoD[('hipDeviceptr_t', 'dst'), ('void*', 'src'), ('size_t', 'sizeBytes')]
    case HIP_API_ID_hipMemcpyHtoD:
      break;
// hipModuleGetGlobal[('hipDeviceptr_t*', 'dptr'), ('size_t*', 'bytes'), ('hipModule_t', 'hmod'), ('const char*', 'name')]
    case HIP_API_ID_hipModuleGetGlobal:
      if (data->args.hipModuleGetGlobal.dptr) data->args.hipModuleGetGlobal.dptr__val = *(data->args.hipModuleGetGlobal.dptr);
      if (data->args.hipModuleGetGlobal.bytes) data->args.hipModuleGetGlobal.bytes__val = *(data->args.hipModuleGetGlobal.bytes);
      if (data->args.hipModuleGetGlobal.name) data->args.hipModuleGetGlobal.name__val = *(data->args.hipModuleGetGlobal.name);
      break;
// hipMemcpyHtoA[('hipArray*', 'dstArray'), ('size_t', 'dstOffset'), ('const void*', 'srcHost'), ('size_t', 'count')]
    case HIP_API_ID_hipMemcpyHtoA:
      if (data->args.hipMemcpyHtoA.dstArray) data->args.hipMemcpyHtoA.dstArray__val = *(data->args.hipMemcpyHtoA.dstArray);
      break;
// hipCtxCreate[('hipCtx_t*', 'ctx'), ('unsigned int', 'flags'), ('hipDevice_t', 'device')]
    case HIP_API_ID_hipCtxCreate:
      if (data->args.hipCtxCreate.ctx) data->args.hipCtxCreate.ctx__val = *(data->args.hipCtxCreate.ctx);
      break;
// hipMemcpy2D[('void*', 'dst'), ('size_t', 'dpitch'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
    case HIP_API_ID_hipMemcpy2D:
      break;
// hipIpcCloseMemHandle[('void*', 'devPtr')]
    case HIP_API_ID_hipIpcCloseMemHandle:
      break;
// hipChooseDevice[('int*', 'device'), ('const hipDeviceProp_t*', 'prop')]
    case HIP_API_ID_hipChooseDevice:
      if (data->args.hipChooseDevice.device) data->args.hipChooseDevice.device__val = *(data->args.hipChooseDevice.device);
      if (data->args.hipChooseDevice.prop) data->args.hipChooseDevice.prop__val = *(data->args.hipChooseDevice.prop);
      break;
// hipDeviceSetSharedMemConfig[('hipSharedMemConfig', 'config')]
    case HIP_API_ID_hipDeviceSetSharedMemConfig:
      break;
// hipMallocMipmappedArray[('hipMipmappedArray_t*', 'mipmappedArray'), ('const hipChannelFormatDesc*', 'desc'), ('hipExtent', 'extent'), ('unsigned int', 'numLevels'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipMallocMipmappedArray:
      if (data->args.hipMallocMipmappedArray.mipmappedArray) data->args.hipMallocMipmappedArray.mipmappedArray__val = *(data->args.hipMallocMipmappedArray.mipmappedArray);
      if (data->args.hipMallocMipmappedArray.desc) data->args.hipMallocMipmappedArray.desc__val = *(data->args.hipMallocMipmappedArray.desc);
      break;
// hipSetupArgument[('const void*', 'arg'), ('size_t', 'size'), ('size_t', 'offset')]
    case HIP_API_ID_hipSetupArgument:
      break;
// hipIpcGetEventHandle[('hipIpcEventHandle_t*', 'handle'), ('hipEvent_t', 'event')]
    case HIP_API_ID_hipIpcGetEventHandle:
      if (data->args.hipIpcGetEventHandle.handle) data->args.hipIpcGetEventHandle.handle__val = *(data->args.hipIpcGetEventHandle.handle);
      break;
// hipFreeArray[('hipArray*', 'array')]
    case HIP_API_ID_hipFreeArray:
      if (data->args.hipFreeArray.array) data->args.hipFreeArray.array__val = *(data->args.hipFreeArray.array);
      break;
// hipCtxSetCacheConfig[('hipFuncCache_t', 'cacheConfig')]
    case HIP_API_ID_hipCtxSetCacheConfig:
      break;
// hipFuncSetCacheConfig[('const void*', 'func'), ('hipFuncCache_t', 'config')]
    case HIP_API_ID_hipFuncSetCacheConfig:
      break;
// hipLaunchKernel[('const void*', 'function_address'), ('dim3', 'numBlocks'), ('dim3', 'dimBlocks'), ('void**', 'args'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipLaunchKernel:
      if (data->args.hipLaunchKernel.args) data->args.hipLaunchKernel.args__val = *(data->args.hipLaunchKernel.args);
      break;
// hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags[('int*', 'numBlocks'), ('hipFunction_t', 'f'), ('int', 'blockSize'), ('size_t', 'dynSharedMemPerBlk'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:
      if (data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks) data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks__val = *(data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks);
      break;
// hipModuleGetTexRef[('textureReference**', 'texRef'), ('hipModule_t', 'hmod'), ('const char*', 'name')]
    case HIP_API_ID_hipModuleGetTexRef:
      if (data->args.hipModuleGetTexRef.texRef) data->args.hipModuleGetTexRef.texRef__val = *(data->args.hipModuleGetTexRef.texRef);
      if (data->args.hipModuleGetTexRef.name) data->args.hipModuleGetTexRef.name__val = *(data->args.hipModuleGetTexRef.name);
      break;
// hipFuncSetAttribute[('const void*', 'func'), ('hipFuncAttribute', 'attr'), ('int', 'value')]
    case HIP_API_ID_hipFuncSetAttribute:
      break;
// hipEventElapsedTime[('float*', 'ms'), ('hipEvent_t', 'start'), ('hipEvent_t', 'stop')]
    case HIP_API_ID_hipEventElapsedTime:
      if (data->args.hipEventElapsedTime.ms) data->args.hipEventElapsedTime.ms__val = *(data->args.hipEventElapsedTime.ms);
      break;
// hipConfigureCall[('dim3', 'gridDim'), ('dim3', 'blockDim'), ('size_t', 'sharedMem'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipConfigureCall:
      break;
// hipMemAdvise[('const void*', 'dev_ptr'), ('size_t', 'count'), ('hipMemoryAdvise', 'advice'), ('int', 'device')]
    case HIP_API_ID_hipMemAdvise:
      break;
// hipMemcpy3DAsync[('const hipMemcpy3DParms*', 'p'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpy3DAsync:
      if (data->args.hipMemcpy3DAsync.p) data->args.hipMemcpy3DAsync.p__val = *(data->args.hipMemcpy3DAsync.p);
      break;
// hipEventDestroy[('hipEvent_t', 'event')]
    case HIP_API_ID_hipEventDestroy:
      break;
// hipCtxPopCurrent[('hipCtx_t*', 'ctx')]
    case HIP_API_ID_hipCtxPopCurrent:
      if (data->args.hipCtxPopCurrent.ctx) data->args.hipCtxPopCurrent.ctx__val = *(data->args.hipCtxPopCurrent.ctx);
      break;
// hipGetSymbolAddress[('void**', 'devPtr'), ('const void*', 'symbol')]
    case HIP_API_ID_hipGetSymbolAddress:
      if (data->args.hipGetSymbolAddress.devPtr) data->args.hipGetSymbolAddress.devPtr__val = *(data->args.hipGetSymbolAddress.devPtr);
      break;
// hipHostGetFlags[('unsigned int*', 'flagsPtr'), ('void*', 'hostPtr')]
    case HIP_API_ID_hipHostGetFlags:
      if (data->args.hipHostGetFlags.flagsPtr) data->args.hipHostGetFlags.flagsPtr__val = *(data->args.hipHostGetFlags.flagsPtr);
      break;
// hipHostMalloc[('void**', 'ptr'), ('size_t', 'size'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipHostMalloc:
      if (data->args.hipHostMalloc.ptr) data->args.hipHostMalloc.ptr__val = *(data->args.hipHostMalloc.ptr);
      break;
// hipCtxSetSharedMemConfig[('hipSharedMemConfig', 'config')]
    case HIP_API_ID_hipCtxSetSharedMemConfig:
      break;
// hipFreeMipmappedArray[('hipMipmappedArray_t', 'mipmappedArray')]
    case HIP_API_ID_hipFreeMipmappedArray:
      break;
// hipMemGetInfo[('size_t*', 'free'), ('size_t*', 'total')]
    case HIP_API_ID_hipMemGetInfo:
      if (data->args.hipMemGetInfo.free) data->args.hipMemGetInfo.free__val = *(data->args.hipMemGetInfo.free);
      if (data->args.hipMemGetInfo.total) data->args.hipMemGetInfo.total__val = *(data->args.hipMemGetInfo.total);
      break;
// hipDeviceReset[]
    case HIP_API_ID_hipDeviceReset:
      break;
// hipMemset[('void*', 'dst'), ('int', 'value'), ('size_t', 'sizeBytes')]
    case HIP_API_ID_hipMemset:
      break;
// hipMemsetD8[('hipDeviceptr_t', 'dest'), ('unsigned char', 'value'), ('size_t', 'count')]
    case HIP_API_ID_hipMemsetD8:
      break;
// hipMemcpyParam2DAsync[('const hip_Memcpy2D*', 'pCopy'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyParam2DAsync:
      if (data->args.hipMemcpyParam2DAsync.pCopy) data->args.hipMemcpyParam2DAsync.pCopy__val = *(data->args.hipMemcpyParam2DAsync.pCopy);
      break;
// hipHostRegister[('void*', 'hostPtr'), ('size_t', 'sizeBytes'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipHostRegister:
      break;
// hipDriverGetVersion[('int*', 'driverVersion')]
    case HIP_API_ID_hipDriverGetVersion:
      if (data->args.hipDriverGetVersion.driverVersion) data->args.hipDriverGetVersion.driverVersion__val = *(data->args.hipDriverGetVersion.driverVersion);
      break;
// hipArray3DCreate[('hipArray**', 'array'), ('const HIP_ARRAY3D_DESCRIPTOR*', 'pAllocateArray')]
    case HIP_API_ID_hipArray3DCreate:
      if (data->args.hipArray3DCreate.array) data->args.hipArray3DCreate.array__val = *(data->args.hipArray3DCreate.array);
      if (data->args.hipArray3DCreate.pAllocateArray) data->args.hipArray3DCreate.pAllocateArray__val = *(data->args.hipArray3DCreate.pAllocateArray);
      break;
// hipIpcOpenMemHandle[('void**', 'devPtr'), ('hipIpcMemHandle_t', 'handle'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipIpcOpenMemHandle:
      if (data->args.hipIpcOpenMemHandle.devPtr) data->args.hipIpcOpenMemHandle.devPtr__val = *(data->args.hipIpcOpenMemHandle.devPtr);
      break;
// hipGetLastError[]
    case HIP_API_ID_hipGetLastError:
      break;
// hipGetDeviceFlags[('unsigned int*', 'flags')]
    case HIP_API_ID_hipGetDeviceFlags:
      if (data->args.hipGetDeviceFlags.flags) data->args.hipGetDeviceFlags.flags__val = *(data->args.hipGetDeviceFlags.flags);
      break;
// hipDeviceGetSharedMemConfig[('hipSharedMemConfig*', 'pConfig')]
    case HIP_API_ID_hipDeviceGetSharedMemConfig:
      if (data->args.hipDeviceGetSharedMemConfig.pConfig) data->args.hipDeviceGetSharedMemConfig.pConfig__val = *(data->args.hipDeviceGetSharedMemConfig.pConfig);
      break;
// hipDrvMemcpy3D[('const HIP_MEMCPY3D*', 'pCopy')]
    case HIP_API_ID_hipDrvMemcpy3D:
      if (data->args.hipDrvMemcpy3D.pCopy) data->args.hipDrvMemcpy3D.pCopy__val = *(data->args.hipDrvMemcpy3D.pCopy);
      break;
// hipMemcpy2DFromArray[('void*', 'dst'), ('size_t', 'dpitch'), ('hipArray_const_t', 'src'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
    case HIP_API_ID_hipMemcpy2DFromArray:
      break;
// hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags[('int*', 'numBlocks'), ('const void*', 'f'), ('int', 'blockSize'), ('size_t', 'dynamicSMemSize'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:
      if (data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks) data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks__val = *(data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks);
      break;
// hipSetDeviceFlags[('unsigned int', 'flags')]
    case HIP_API_ID_hipSetDeviceFlags:
      break;
// hipHccModuleLaunchKernel[('hipFunction_t', 'f'), ('unsigned int', 'globalWorkSizeX'), ('unsigned int', 'globalWorkSizeY'), ('unsigned int', 'globalWorkSizeZ'), ('unsigned int', 'blockDimX'), ('unsigned int', 'blockDimY'), ('unsigned int', 'blockDimZ'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'hStream'), ('void**', 'kernelParams'), ('void**', 'extra'), ('hipEvent_t', 'startEvent'), ('hipEvent_t', 'stopEvent')]
    case HIP_API_ID_hipHccModuleLaunchKernel:
      if (data->args.hipHccModuleLaunchKernel.kernelParams) data->args.hipHccModuleLaunchKernel.kernelParams__val = *(data->args.hipHccModuleLaunchKernel.kernelParams);
      if (data->args.hipHccModuleLaunchKernel.extra) data->args.hipHccModuleLaunchKernel.extra__val = *(data->args.hipHccModuleLaunchKernel.extra);
      break;
// hipFree[('void*', 'ptr')]
    case HIP_API_ID_hipFree:
      break;
// hipOccupancyMaxPotentialBlockSize[('int*', 'gridSize'), ('int*', 'blockSize'), ('const void*', 'f'), ('size_t', 'dynSharedMemPerBlk'), ('int', 'blockSizeLimit')]
    case HIP_API_ID_hipOccupancyMaxPotentialBlockSize:
      if (data->args.hipOccupancyMaxPotentialBlockSize.gridSize) data->args.hipOccupancyMaxPotentialBlockSize.gridSize__val = *(data->args.hipOccupancyMaxPotentialBlockSize.gridSize);
      if (data->args.hipOccupancyMaxPotentialBlockSize.blockSize) data->args.hipOccupancyMaxPotentialBlockSize.blockSize__val = *(data->args.hipOccupancyMaxPotentialBlockSize.blockSize);
      break;
// hipDeviceGetAttribute[('int*', 'pi'), ('hipDeviceAttribute_t', 'attr'), ('int', 'deviceId')]
    case HIP_API_ID_hipDeviceGetAttribute:
      if (data->args.hipDeviceGetAttribute.pi) data->args.hipDeviceGetAttribute.pi__val = *(data->args.hipDeviceGetAttribute.pi);
      break;
// hipDeviceComputeCapability[('int*', 'major'), ('int*', 'minor'), ('hipDevice_t', 'device')]
    case HIP_API_ID_hipDeviceComputeCapability:
      if (data->args.hipDeviceComputeCapability.major) data->args.hipDeviceComputeCapability.major__val = *(data->args.hipDeviceComputeCapability.major);
      if (data->args.hipDeviceComputeCapability.minor) data->args.hipDeviceComputeCapability.minor__val = *(data->args.hipDeviceComputeCapability.minor);
      break;
// hipCtxDisablePeerAccess[('hipCtx_t', 'peerCtx')]
    case HIP_API_ID_hipCtxDisablePeerAccess:
      break;
// hipMallocManaged[('void**', 'dev_ptr'), ('size_t', 'size'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipMallocManaged:
      if (data->args.hipMallocManaged.dev_ptr) data->args.hipMallocManaged.dev_ptr__val = *(data->args.hipMallocManaged.dev_ptr);
      break;
// hipDeviceGetByPCIBusId[('int*', 'device'), ('const char*', 'pciBusId')]
    case HIP_API_ID_hipDeviceGetByPCIBusId:
      if (data->args.hipDeviceGetByPCIBusId.device) data->args.hipDeviceGetByPCIBusId.device__val = *(data->args.hipDeviceGetByPCIBusId.device);
      if (data->args.hipDeviceGetByPCIBusId.pciBusId) data->args.hipDeviceGetByPCIBusId.pciBusId__val = *(data->args.hipDeviceGetByPCIBusId.pciBusId);
      break;
// hipIpcGetMemHandle[('hipIpcMemHandle_t*', 'handle'), ('void*', 'devPtr')]
    case HIP_API_ID_hipIpcGetMemHandle:
      if (data->args.hipIpcGetMemHandle.handle) data->args.hipIpcGetMemHandle.handle__val = *(data->args.hipIpcGetMemHandle.handle);
      break;
// hipMemcpyHtoDAsync[('hipDeviceptr_t', 'dst'), ('void*', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyHtoDAsync:
      break;
// hipCtxGetDevice[('hipDevice_t*', 'device')]
    case HIP_API_ID_hipCtxGetDevice:
      if (data->args.hipCtxGetDevice.device) data->args.hipCtxGetDevice.device__val = *(data->args.hipCtxGetDevice.device);
      break;
// hipMemcpyDtoD[('hipDeviceptr_t', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes')]
    case HIP_API_ID_hipMemcpyDtoD:
      break;
// hipModuleLoadData[('hipModule_t*', 'module'), ('const void*', 'image')]
    case HIP_API_ID_hipModuleLoadData:
      if (data->args.hipModuleLoadData.module) data->args.hipModuleLoadData.module__val = *(data->args.hipModuleLoadData.module);
      break;
// hipDevicePrimaryCtxRelease[('hipDevice_t', 'dev')]
    case HIP_API_ID_hipDevicePrimaryCtxRelease:
      break;
// hipOccupancyMaxActiveBlocksPerMultiprocessor[('int*', 'numBlocks'), ('const void*', 'f'), ('int', 'blockSize'), ('size_t', 'dynamicSMemSize')]
    case HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor:
      if (data->args.hipOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks) data->args.hipOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks__val = *(data->args.hipOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks);
      break;
// hipCtxSetCurrent[('hipCtx_t', 'ctx')]
    case HIP_API_ID_hipCtxSetCurrent:
      break;
// hipGetErrorString[]
    case HIP_API_ID_hipGetErrorString:
      break;
// hipStreamCreate[('hipStream_t*', 'stream')]
    case HIP_API_ID_hipStreamCreate:
      if (data->args.hipStreamCreate.stream) data->args.hipStreamCreate.stream__val = *(data->args.hipStreamCreate.stream);
      break;
// hipDevicePrimaryCtxRetain[('hipCtx_t*', 'pctx'), ('hipDevice_t', 'dev')]
    case HIP_API_ID_hipDevicePrimaryCtxRetain:
      if (data->args.hipDevicePrimaryCtxRetain.pctx) data->args.hipDevicePrimaryCtxRetain.pctx__val = *(data->args.hipDevicePrimaryCtxRetain.pctx);
      break;
// hipDeviceGet[('hipDevice_t*', 'device'), ('int', 'ordinal')]
    case HIP_API_ID_hipDeviceGet:
      if (data->args.hipDeviceGet.device) data->args.hipDeviceGet.device__val = *(data->args.hipDeviceGet.device);
      break;
// hipStreamCreateWithFlags[('hipStream_t*', 'stream'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipStreamCreateWithFlags:
      if (data->args.hipStreamCreateWithFlags.stream) data->args.hipStreamCreateWithFlags.stream__val = *(data->args.hipStreamCreateWithFlags.stream);
      break;
// hipMemcpyFromArray[('void*', 'dst'), ('hipArray_const_t', 'srcArray'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'count'), ('hipMemcpyKind', 'kind')]
    case HIP_API_ID_hipMemcpyFromArray:
      break;
// hipMemcpy2DAsync[('void*', 'dst'), ('size_t', 'dpitch'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpy2DAsync:
      break;
// hipFuncGetAttributes[('hipFuncAttributes*', 'attr'), ('const void*', 'func')]
    case HIP_API_ID_hipFuncGetAttributes:
      if (data->args.hipFuncGetAttributes.attr) data->args.hipFuncGetAttributes.attr__val = *(data->args.hipFuncGetAttributes.attr);
      break;
// hipGetSymbolSize[('size_t*', 'size'), ('const void*', 'symbol')]
    case HIP_API_ID_hipGetSymbolSize:
      if (data->args.hipGetSymbolSize.size) data->args.hipGetSymbolSize.size__val = *(data->args.hipGetSymbolSize.size);
      break;
// hipHostFree[('void*', 'ptr')]
    case HIP_API_ID_hipHostFree:
      break;
// hipEventCreateWithFlags[('hipEvent_t*', 'event'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipEventCreateWithFlags:
      if (data->args.hipEventCreateWithFlags.event) data->args.hipEventCreateWithFlags.event__val = *(data->args.hipEventCreateWithFlags.event);
      break;
// hipStreamQuery[('hipStream_t', 'stream')]
    case HIP_API_ID_hipStreamQuery:
      break;
// hipMemcpy3D[('const hipMemcpy3DParms*', 'p')]
    case HIP_API_ID_hipMemcpy3D:
      if (data->args.hipMemcpy3D.p) data->args.hipMemcpy3D.p__val = *(data->args.hipMemcpy3D.p);
      break;
// hipMemcpyToSymbol[('const void*', 'symbol'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind')]
    case HIP_API_ID_hipMemcpyToSymbol:
      break;
// hipMemcpy[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind')]
    case HIP_API_ID_hipMemcpy:
      break;
// hipPeekAtLastError[]
    case HIP_API_ID_hipPeekAtLastError:
      break;
// hipExtLaunchMultiKernelMultiDevice[('hipLaunchParams*', 'launchParamsList'), ('int', 'numDevices'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
      if (data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList) data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList__val = *(data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList);
      break;
// hipHostAlloc[('void**', 'ptr'), ('size_t', 'size'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipHostAlloc:
      if (data->args.hipHostAlloc.ptr) data->args.hipHostAlloc.ptr__val = *(data->args.hipHostAlloc.ptr);
      break;
// hipStreamAddCallback[('hipStream_t', 'stream'), ('hipStreamCallback_t', 'callback'), ('void*', 'userData'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipStreamAddCallback:
      break;
// hipMemcpyToArray[('hipArray*', 'dst'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('const void*', 'src'), ('size_t', 'count'), ('hipMemcpyKind', 'kind')]
    case HIP_API_ID_hipMemcpyToArray:
      if (data->args.hipMemcpyToArray.dst) data->args.hipMemcpyToArray.dst__val = *(data->args.hipMemcpyToArray.dst);
      break;
// hipMemsetD32[('hipDeviceptr_t', 'dest'), ('int', 'value'), ('size_t', 'count')]
    case HIP_API_ID_hipMemsetD32:
      break;
// hipExtModuleLaunchKernel[('hipFunction_t', 'f'), ('unsigned int', 'globalWorkSizeX'), ('unsigned int', 'globalWorkSizeY'), ('unsigned int', 'globalWorkSizeZ'), ('unsigned int', 'localWorkSizeX'), ('unsigned int', 'localWorkSizeY'), ('unsigned int', 'localWorkSizeZ'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'hStream'), ('void**', 'kernelParams'), ('void**', 'extra'), ('hipEvent_t', 'startEvent'), ('hipEvent_t', 'stopEvent'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipExtModuleLaunchKernel:
      if (data->args.hipExtModuleLaunchKernel.kernelParams) data->args.hipExtModuleLaunchKernel.kernelParams__val = *(data->args.hipExtModuleLaunchKernel.kernelParams);
      if (data->args.hipExtModuleLaunchKernel.extra) data->args.hipExtModuleLaunchKernel.extra__val = *(data->args.hipExtModuleLaunchKernel.extra);
      break;
// hipDeviceSynchronize[]
    case HIP_API_ID_hipDeviceSynchronize:
      break;
// hipDeviceGetCacheConfig[('hipFuncCache_t*', 'cacheConfig')]
    case HIP_API_ID_hipDeviceGetCacheConfig:
      if (data->args.hipDeviceGetCacheConfig.cacheConfig) data->args.hipDeviceGetCacheConfig.cacheConfig__val = *(data->args.hipDeviceGetCacheConfig.cacheConfig);
      break;
// hipMalloc3D[('hipPitchedPtr*', 'pitchedDevPtr'), ('hipExtent', 'extent')]
    case HIP_API_ID_hipMalloc3D:
      if (data->args.hipMalloc3D.pitchedDevPtr) data->args.hipMalloc3D.pitchedDevPtr__val = *(data->args.hipMalloc3D.pitchedDevPtr);
      break;
// hipPointerGetAttributes[('hipPointerAttribute_t*', 'attributes'), ('const void*', 'ptr')]
    case HIP_API_ID_hipPointerGetAttributes:
      if (data->args.hipPointerGetAttributes.attributes) data->args.hipPointerGetAttributes.attributes__val = *(data->args.hipPointerGetAttributes.attributes);
      break;
// hipMemsetAsync[('void*', 'dst'), ('int', 'value'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemsetAsync:
      break;
// hipDeviceGetName[('char*', 'name'), ('int', 'len'), ('hipDevice_t', 'device')]
    case HIP_API_ID_hipDeviceGetName:
      data->args.hipDeviceGetName.name = (data->args.hipDeviceGetName.name) ? strdup(data->args.hipDeviceGetName.name) : NULL;
      break;
// hipModuleOccupancyMaxPotentialBlockSizeWithFlags[('int*', 'gridSize'), ('int*', 'blockSize'), ('hipFunction_t', 'f'), ('size_t', 'dynSharedMemPerBlk'), ('int', 'blockSizeLimit'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags:
      if (data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize) data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize__val = *(data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize);
      if (data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize) data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize__val = *(data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize);
      break;
// hipCtxPushCurrent[('hipCtx_t', 'ctx')]
    case HIP_API_ID_hipCtxPushCurrent:
      break;
// hipMemcpyPeer[('void*', 'dst'), ('int', 'dstDeviceId'), ('const void*', 'src'), ('int', 'srcDeviceId'), ('size_t', 'sizeBytes')]
    case HIP_API_ID_hipMemcpyPeer:
      break;
// hipEventSynchronize[('hipEvent_t', 'event')]
    case HIP_API_ID_hipEventSynchronize:
      break;
// hipMemcpyDtoDAsync[('hipDeviceptr_t', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyDtoDAsync:
      break;
// hipProfilerStart[]
    case HIP_API_ID_hipProfilerStart:
      break;
// hipExtMallocWithFlags[('void**', 'ptr'), ('size_t', 'sizeBytes'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipExtMallocWithFlags:
      if (data->args.hipExtMallocWithFlags.ptr) data->args.hipExtMallocWithFlags.ptr__val = *(data->args.hipExtMallocWithFlags.ptr);
      break;
// hipCtxEnablePeerAccess[('hipCtx_t', 'peerCtx'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipCtxEnablePeerAccess:
      break;
// hipMemAllocHost[('void**', 'ptr'), ('size_t', 'size')]
    case HIP_API_ID_hipMemAllocHost:
      if (data->args.hipMemAllocHost.ptr) data->args.hipMemAllocHost.ptr__val = *(data->args.hipMemAllocHost.ptr);
      break;
// hipMemcpyDtoHAsync[('void*', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyDtoHAsync:
      break;
// hipModuleLaunchKernel[('hipFunction_t', 'f'), ('unsigned int', 'gridDimX'), ('unsigned int', 'gridDimY'), ('unsigned int', 'gridDimZ'), ('unsigned int', 'blockDimX'), ('unsigned int', 'blockDimY'), ('unsigned int', 'blockDimZ'), ('unsigned int', 'sharedMemBytes'), ('hipStream_t', 'stream'), ('void**', 'kernelParams'), ('void**', 'extra')]
    case HIP_API_ID_hipModuleLaunchKernel:
      if (data->args.hipModuleLaunchKernel.kernelParams) data->args.hipModuleLaunchKernel.kernelParams__val = *(data->args.hipModuleLaunchKernel.kernelParams);
      if (data->args.hipModuleLaunchKernel.extra) data->args.hipModuleLaunchKernel.extra__val = *(data->args.hipModuleLaunchKernel.extra);
      break;
// hipMemAllocPitch[('hipDeviceptr_t*', 'dptr'), ('size_t*', 'pitch'), ('size_t', 'widthInBytes'), ('size_t', 'height'), ('unsigned int', 'elementSizeBytes')]
    case HIP_API_ID_hipMemAllocPitch:
      if (data->args.hipMemAllocPitch.dptr) data->args.hipMemAllocPitch.dptr__val = *(data->args.hipMemAllocPitch.dptr);
      if (data->args.hipMemAllocPitch.pitch) data->args.hipMemAllocPitch.pitch__val = *(data->args.hipMemAllocPitch.pitch);
      break;
// hipExtLaunchKernel[('const void*', 'function_address'), ('dim3', 'numBlocks'), ('dim3', 'dimBlocks'), ('void**', 'args'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'stream'), ('hipEvent_t', 'startEvent'), ('hipEvent_t', 'stopEvent'), ('int', 'flags')]
    case HIP_API_ID_hipExtLaunchKernel:
      if (data->args.hipExtLaunchKernel.args) data->args.hipExtLaunchKernel.args__val = *(data->args.hipExtLaunchKernel.args);
      break;
// hipMemcpy2DFromArrayAsync[('void*', 'dst'), ('size_t', 'dpitch'), ('hipArray_const_t', 'src'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpy2DFromArrayAsync:
      break;
// hipDeviceGetLimit[('size_t*', 'pValue'), ('hipLimit_t', 'limit')]
    case HIP_API_ID_hipDeviceGetLimit:
      if (data->args.hipDeviceGetLimit.pValue) data->args.hipDeviceGetLimit.pValue__val = *(data->args.hipDeviceGetLimit.pValue);
      break;
// hipModuleLoadDataEx[('hipModule_t*', 'module'), ('const void*', 'image'), ('unsigned int', 'numOptions'), ('hipJitOption*', 'options'), ('void**', 'optionsValues')]
    case HIP_API_ID_hipModuleLoadDataEx:
      if (data->args.hipModuleLoadDataEx.module) data->args.hipModuleLoadDataEx.module__val = *(data->args.hipModuleLoadDataEx.module);
      if (data->args.hipModuleLoadDataEx.options) data->args.hipModuleLoadDataEx.options__val = *(data->args.hipModuleLoadDataEx.options);
      if (data->args.hipModuleLoadDataEx.optionsValues) data->args.hipModuleLoadDataEx.optionsValues__val = *(data->args.hipModuleLoadDataEx.optionsValues);
      break;
// hipRuntimeGetVersion[('int*', 'runtimeVersion')]
    case HIP_API_ID_hipRuntimeGetVersion:
      if (data->args.hipRuntimeGetVersion.runtimeVersion) data->args.hipRuntimeGetVersion.runtimeVersion__val = *(data->args.hipRuntimeGetVersion.runtimeVersion);
      break;
// hipMemRangeGetAttribute[('void*', 'data'), ('size_t', 'data_size'), ('hipMemRangeAttribute', 'attribute'), ('const void*', 'dev_ptr'), ('size_t', 'count')]
    case HIP_API_ID_hipMemRangeGetAttribute:
      break;
// hipDeviceGetP2PAttribute[('int*', 'value'), ('hipDeviceP2PAttr', 'attr'), ('int', 'srcDevice'), ('int', 'dstDevice')]
    case HIP_API_ID_hipDeviceGetP2PAttribute:
      if (data->args.hipDeviceGetP2PAttribute.value) data->args.hipDeviceGetP2PAttribute.value__val = *(data->args.hipDeviceGetP2PAttribute.value);
      break;
// hipMemcpyPeerAsync[('void*', 'dst'), ('int', 'dstDeviceId'), ('const void*', 'src'), ('int', 'srcDevice'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyPeerAsync:
      break;
// hipGetDeviceProperties[('hipDeviceProp_t*', 'props'), ('hipDevice_t', 'device')]
    case HIP_API_ID_hipGetDeviceProperties:
      if (data->args.hipGetDeviceProperties.props) data->args.hipGetDeviceProperties.props__val = *(data->args.hipGetDeviceProperties.props);
      break;
// hipMemcpyDtoH[('void*', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes')]
    case HIP_API_ID_hipMemcpyDtoH:
      break;
// hipMemcpyWithStream[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemcpyWithStream:
      break;
// hipDeviceTotalMem[('size_t*', 'bytes'), ('hipDevice_t', 'device')]
    case HIP_API_ID_hipDeviceTotalMem:
      if (data->args.hipDeviceTotalMem.bytes) data->args.hipDeviceTotalMem.bytes__val = *(data->args.hipDeviceTotalMem.bytes);
      break;
// hipHostGetDevicePointer[('void**', 'devPtr'), ('void*', 'hstPtr'), ('unsigned int', 'flags')]
    case HIP_API_ID_hipHostGetDevicePointer:
      if (data->args.hipHostGetDevicePointer.devPtr) data->args.hipHostGetDevicePointer.devPtr__val = *(data->args.hipHostGetDevicePointer.devPtr);
      break;
// hipMemRangeGetAttributes[('void**', 'data'), ('size_t*', 'data_sizes'), ('hipMemRangeAttribute*', 'attributes'), ('size_t', 'num_attributes'), ('const void*', 'dev_ptr'), ('size_t', 'count')]
    case HIP_API_ID_hipMemRangeGetAttributes:
      if (data->args.hipMemRangeGetAttributes.data) data->args.hipMemRangeGetAttributes.data__val = *(data->args.hipMemRangeGetAttributes.data);
      if (data->args.hipMemRangeGetAttributes.data_sizes) data->args.hipMemRangeGetAttributes.data_sizes__val = *(data->args.hipMemRangeGetAttributes.data_sizes);
      if (data->args.hipMemRangeGetAttributes.attributes) data->args.hipMemRangeGetAttributes.attributes__val = *(data->args.hipMemRangeGetAttributes.attributes);
      break;
// hipMemcpyParam2D[('const hip_Memcpy2D*', 'pCopy')]
    case HIP_API_ID_hipMemcpyParam2D:
      if (data->args.hipMemcpyParam2D.pCopy) data->args.hipMemcpyParam2D.pCopy__val = *(data->args.hipMemcpyParam2D.pCopy);
      break;
// hipDevicePrimaryCtxReset[('hipDevice_t', 'dev')]
    case HIP_API_ID_hipDevicePrimaryCtxReset:
      break;
// hipGetMipmappedArrayLevel[('hipArray_t*', 'levelArray'), ('hipMipmappedArray_const_t', 'mipmappedArray'), ('unsigned int', 'level')]
    case HIP_API_ID_hipGetMipmappedArrayLevel:
      if (data->args.hipGetMipmappedArrayLevel.levelArray) data->args.hipGetMipmappedArrayLevel.levelArray__val = *(data->args.hipGetMipmappedArrayLevel.levelArray);
      break;
// hipMemsetD32Async[('hipDeviceptr_t', 'dst'), ('int', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
    case HIP_API_ID_hipMemsetD32Async:
      break;
// hipGetDevice[('int*', 'deviceId')]
    case HIP_API_ID_hipGetDevice:
      if (data->args.hipGetDevice.deviceId) data->args.hipGetDevice.deviceId__val = *(data->args.hipGetDevice.deviceId);
      break;
// hipGetDeviceCount[('int*', 'count')]
    case HIP_API_ID_hipGetDeviceCount:
      if (data->args.hipGetDeviceCount.count) data->args.hipGetDeviceCount.count__val = *(data->args.hipGetDeviceCount.count);
      break;
// hipIpcOpenEventHandle[('hipEvent_t*', 'event'), ('hipIpcEventHandle_t', 'handle')]
    case HIP_API_ID_hipIpcOpenEventHandle:
      if (data->args.hipIpcOpenEventHandle.event) data->args.hipIpcOpenEventHandle.event__val = *(data->args.hipIpcOpenEventHandle.event);
      break;
    default: break;
  };
}

#include <sstream>
#include <string>
// HIP API string method, method name and parameters
static inline const char* hipApiString(hip_api_id_t id, const hip_api_data_t* data) {
  std::ostringstream oss;
  switch (id) {
    case HIP_API_ID_hipDrvMemcpy3DAsync:
      oss << "hipDrvMemcpy3DAsync(";
      if (data->args.hipDrvMemcpy3DAsync.pCopy == NULL) oss << "pCopy=NULL";
      else oss << "pCopy=" << data->args.hipDrvMemcpy3DAsync.pCopy__val;
      oss << ", stream=" << data->args.hipDrvMemcpy3DAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceEnablePeerAccess:
      oss << "hipDeviceEnablePeerAccess(";
      oss << "peerDeviceId=" << data->args.hipDeviceEnablePeerAccess.peerDeviceId;
      oss << ", flags=" << data->args.hipDeviceEnablePeerAccess.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipFuncSetSharedMemConfig:
      oss << "hipFuncSetSharedMemConfig(";
      oss << "func=" << data->args.hipFuncSetSharedMemConfig.func;
      oss << ", config=" << data->args.hipFuncSetSharedMemConfig.config;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyToSymbolAsync:
      oss << "hipMemcpyToSymbolAsync(";
      oss << "symbol=" << data->args.hipMemcpyToSymbolAsync.symbol;
      oss << ", src=" << data->args.hipMemcpyToSymbolAsync.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyToSymbolAsync.sizeBytes;
      oss << ", offset=" << data->args.hipMemcpyToSymbolAsync.offset;
      oss << ", kind=" << data->args.hipMemcpyToSymbolAsync.kind;
      oss << ", stream=" << data->args.hipMemcpyToSymbolAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipMallocPitch:
      oss << "hipMallocPitch(";
      if (data->args.hipMallocPitch.ptr == NULL) oss << "ptr=NULL";
      else oss << "ptr=" << data->args.hipMallocPitch.ptr__val;
      if (data->args.hipMallocPitch.pitch == NULL) oss << ", pitch=NULL";
      else oss << ", pitch=" << data->args.hipMallocPitch.pitch__val;
      oss << ", width=" << data->args.hipMallocPitch.width;
      oss << ", height=" << data->args.hipMallocPitch.height;
      oss << ")";
    break;
    case HIP_API_ID_hipMalloc:
      oss << "hipMalloc(";
      if (data->args.hipMalloc.ptr == NULL) oss << "ptr=NULL";
      else oss << "ptr=" << data->args.hipMalloc.ptr__val;
      oss << ", size=" << data->args.hipMalloc.size;
      oss << ")";
    break;
    case HIP_API_ID_hipMemsetD16:
      oss << "hipMemsetD16(";
      oss << "dest=" << data->args.hipMemsetD16.dest;
      oss << ", value=" << data->args.hipMemsetD16.value;
      oss << ", count=" << data->args.hipMemsetD16.count;
      oss << ")";
    break;
    case HIP_API_ID_hipExtStreamGetCUMask:
      oss << "hipExtStreamGetCUMask(";
      oss << "stream=" << data->args.hipExtStreamGetCUMask.stream;
      oss << ", cuMaskSize=" << data->args.hipExtStreamGetCUMask.cuMaskSize;
      if (data->args.hipExtStreamGetCUMask.cuMask == NULL) oss << ", cuMask=NULL";
      else oss << ", cuMask=" << data->args.hipExtStreamGetCUMask.cuMask__val;
      oss << ")";
    break;
    case HIP_API_ID_hipEventRecord:
      oss << "hipEventRecord(";
      oss << "event=" << data->args.hipEventRecord.event;
      oss << ", stream=" << data->args.hipEventRecord.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxSynchronize:
      oss << "hipCtxSynchronize(";
      oss << ")";
    break;
    case HIP_API_ID_hipSetDevice:
      oss << "hipSetDevice(";
      oss << "deviceId=" << data->args.hipSetDevice.deviceId;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxGetApiVersion:
      oss << "hipCtxGetApiVersion(";
      oss << "ctx=" << data->args.hipCtxGetApiVersion.ctx;
      if (data->args.hipCtxGetApiVersion.apiVersion == NULL) oss << ", apiVersion=NULL";
      else oss << ", apiVersion=" << data->args.hipCtxGetApiVersion.apiVersion__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyFromSymbolAsync:
      oss << "hipMemcpyFromSymbolAsync(";
      oss << "dst=" << data->args.hipMemcpyFromSymbolAsync.dst;
      oss << ", symbol=" << data->args.hipMemcpyFromSymbolAsync.symbol;
      oss << ", sizeBytes=" << data->args.hipMemcpyFromSymbolAsync.sizeBytes;
      oss << ", offset=" << data->args.hipMemcpyFromSymbolAsync.offset;
      oss << ", kind=" << data->args.hipMemcpyFromSymbolAsync.kind;
      oss << ", stream=" << data->args.hipMemcpyFromSymbolAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipExtGetLinkTypeAndHopCount:
      oss << "hipExtGetLinkTypeAndHopCount(";
      oss << "device1=" << data->args.hipExtGetLinkTypeAndHopCount.device1;
      oss << ", device2=" << data->args.hipExtGetLinkTypeAndHopCount.device2;
      if (data->args.hipExtGetLinkTypeAndHopCount.linktype == NULL) oss << ", linktype=NULL";
      else oss << ", linktype=" << data->args.hipExtGetLinkTypeAndHopCount.linktype__val;
      if (data->args.hipExtGetLinkTypeAndHopCount.hopcount == NULL) oss << ", hopcount=NULL";
      else oss << ", hopcount=" << data->args.hipExtGetLinkTypeAndHopCount.hopcount__val;
      oss << ")";
    break;
    case HIP_API_ID___hipPopCallConfiguration:
      oss << "__hipPopCallConfiguration(";
      if (data->args.__hipPopCallConfiguration.gridDim == NULL) oss << "gridDim=NULL";
      else oss << "gridDim=" << data->args.__hipPopCallConfiguration.gridDim__val;
      if (data->args.__hipPopCallConfiguration.blockDim == NULL) oss << ", blockDim=NULL";
      else oss << ", blockDim=" << data->args.__hipPopCallConfiguration.blockDim__val;
      if (data->args.__hipPopCallConfiguration.sharedMem == NULL) oss << ", sharedMem=NULL";
      else oss << ", sharedMem=" << data->args.__hipPopCallConfiguration.sharedMem__val;
      if (data->args.__hipPopCallConfiguration.stream == NULL) oss << ", stream=NULL";
      else oss << ", stream=" << data->args.__hipPopCallConfiguration.stream__val;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor:
      oss << "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(";
      if (data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks == NULL) oss << "numBlocks=NULL";
      else oss << "numBlocks=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks__val;
      oss << ", f=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.f;
      oss << ", blockSize=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.blockSize;
      oss << ", dynSharedMemPerBlk=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.dynSharedMemPerBlk;
      oss << ")";
    break;
    case HIP_API_ID_hipMemset3D:
      oss << "hipMemset3D(";
      oss << "pitchedDevPtr=" << data->args.hipMemset3D.pitchedDevPtr;
      oss << ", value=" << data->args.hipMemset3D.value;
      oss << ", extent=" << data->args.hipMemset3D.extent;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamCreateWithPriority:
      oss << "hipStreamCreateWithPriority(";
      if (data->args.hipStreamCreateWithPriority.stream == NULL) oss << "stream=NULL";
      else oss << "stream=" << data->args.hipStreamCreateWithPriority.stream__val;
      oss << ", flags=" << data->args.hipStreamCreateWithPriority.flags;
      oss << ", priority=" << data->args.hipStreamCreateWithPriority.priority;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpy2DToArray:
      oss << "hipMemcpy2DToArray(";
      if (data->args.hipMemcpy2DToArray.dst == NULL) oss << "dst=NULL";
      else oss << "dst=" << data->args.hipMemcpy2DToArray.dst__val;
      oss << ", wOffset=" << data->args.hipMemcpy2DToArray.wOffset;
      oss << ", hOffset=" << data->args.hipMemcpy2DToArray.hOffset;
      oss << ", src=" << data->args.hipMemcpy2DToArray.src;
      oss << ", spitch=" << data->args.hipMemcpy2DToArray.spitch;
      oss << ", width=" << data->args.hipMemcpy2DToArray.width;
      oss << ", height=" << data->args.hipMemcpy2DToArray.height;
      oss << ", kind=" << data->args.hipMemcpy2DToArray.kind;
      oss << ")";
    break;
    case HIP_API_ID_hipMemsetD8Async:
      oss << "hipMemsetD8Async(";
      oss << "dest=" << data->args.hipMemsetD8Async.dest;
      oss << ", value=" << data->args.hipMemsetD8Async.value;
      oss << ", count=" << data->args.hipMemsetD8Async.count;
      oss << ", stream=" << data->args.hipMemsetD8Async.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxGetCacheConfig:
      oss << "hipCtxGetCacheConfig(";
      if (data->args.hipCtxGetCacheConfig.cacheConfig == NULL) oss << "cacheConfig=NULL";
      else oss << "cacheConfig=" << data->args.hipCtxGetCacheConfig.cacheConfig__val;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleGetFunction:
      oss << "hipModuleGetFunction(";
      if (data->args.hipModuleGetFunction.function == NULL) oss << "function=NULL";
      else oss << "function=" << data->args.hipModuleGetFunction.function__val;
      oss << ", module=" << data->args.hipModuleGetFunction.module;
      if (data->args.hipModuleGetFunction.kname == NULL) oss << ", kname=NULL";
      else oss << ", kname=" << data->args.hipModuleGetFunction.kname__val;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamWaitEvent:
      oss << "hipStreamWaitEvent(";
      oss << "stream=" << data->args.hipStreamWaitEvent.stream;
      oss << ", event=" << data->args.hipStreamWaitEvent.event;
      oss << ", flags=" << data->args.hipStreamWaitEvent.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetStreamPriorityRange:
      oss << "hipDeviceGetStreamPriorityRange(";
      if (data->args.hipDeviceGetStreamPriorityRange.leastPriority == NULL) oss << "leastPriority=NULL";
      else oss << "leastPriority=" << data->args.hipDeviceGetStreamPriorityRange.leastPriority__val;
      if (data->args.hipDeviceGetStreamPriorityRange.greatestPriority == NULL) oss << ", greatestPriority=NULL";
      else oss << ", greatestPriority=" << data->args.hipDeviceGetStreamPriorityRange.greatestPriority__val;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleLoad:
      oss << "hipModuleLoad(";
      if (data->args.hipModuleLoad.module == NULL) oss << "module=NULL";
      else oss << "module=" << data->args.hipModuleLoad.module__val;
      if (data->args.hipModuleLoad.fname == NULL) oss << ", fname=NULL";
      else oss << ", fname=" << data->args.hipModuleLoad.fname__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxSetFlags:
      oss << "hipDevicePrimaryCtxSetFlags(";
      oss << "dev=" << data->args.hipDevicePrimaryCtxSetFlags.dev;
      oss << ", flags=" << data->args.hipDevicePrimaryCtxSetFlags.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipLaunchCooperativeKernel:
      oss << "hipLaunchCooperativeKernel(";
      oss << "f=" << data->args.hipLaunchCooperativeKernel.f;
      oss << ", gridDim=" << data->args.hipLaunchCooperativeKernel.gridDim;
      oss << ", blockDimX=" << data->args.hipLaunchCooperativeKernel.blockDimX;
      if (data->args.hipLaunchCooperativeKernel.kernelParams == NULL) oss << ", kernelParams=NULL";
      else oss << ", kernelParams=" << data->args.hipLaunchCooperativeKernel.kernelParams__val;
      oss << ", sharedMemBytes=" << data->args.hipLaunchCooperativeKernel.sharedMemBytes;
      oss << ", stream=" << data->args.hipLaunchCooperativeKernel.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
      oss << "hipLaunchCooperativeKernelMultiDevice(";
      if (data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList == NULL) oss << "launchParamsList=NULL";
      else oss << "launchParamsList=" << data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val;
      oss << ", numDevices=" << data->args.hipLaunchCooperativeKernelMultiDevice.numDevices;
      oss << ", flags=" << data->args.hipLaunchCooperativeKernelMultiDevice.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyAsync:
      oss << "hipMemcpyAsync(";
      oss << "dst=" << data->args.hipMemcpyAsync.dst;
      oss << ", src=" << data->args.hipMemcpyAsync.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyAsync.sizeBytes;
      oss << ", kind=" << data->args.hipMemcpyAsync.kind;
      oss << ", stream=" << data->args.hipMemcpyAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipMalloc3DArray:
      oss << "hipMalloc3DArray(";
      if (data->args.hipMalloc3DArray.array == NULL) oss << "array=NULL";
      else oss << "array=" << data->args.hipMalloc3DArray.array__val;
      if (data->args.hipMalloc3DArray.desc == NULL) oss << ", desc=NULL";
      else oss << ", desc=" << data->args.hipMalloc3DArray.desc__val;
      oss << ", extent=" << data->args.hipMalloc3DArray.extent;
      oss << ", flags=" << data->args.hipMalloc3DArray.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipMallocHost:
      oss << "hipMallocHost(";
      if (data->args.hipMallocHost.ptr == NULL) oss << "ptr=NULL";
      else oss << "ptr=" << data->args.hipMallocHost.ptr__val;
      oss << ", size=" << data->args.hipMallocHost.size;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxGetCurrent:
      oss << "hipCtxGetCurrent(";
      if (data->args.hipCtxGetCurrent.ctx == NULL) oss << "ctx=NULL";
      else oss << "ctx=" << data->args.hipCtxGetCurrent.ctx__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxGetState:
      oss << "hipDevicePrimaryCtxGetState(";
      oss << "dev=" << data->args.hipDevicePrimaryCtxGetState.dev;
      if (data->args.hipDevicePrimaryCtxGetState.flags == NULL) oss << ", flags=NULL";
      else oss << ", flags=" << data->args.hipDevicePrimaryCtxGetState.flags__val;
      if (data->args.hipDevicePrimaryCtxGetState.active == NULL) oss << ", active=NULL";
      else oss << ", active=" << data->args.hipDevicePrimaryCtxGetState.active__val;
      oss << ")";
    break;
    case HIP_API_ID_hipEventQuery:
      oss << "hipEventQuery(";
      oss << "event=" << data->args.hipEventQuery.event;
      oss << ")";
    break;
    case HIP_API_ID_hipEventCreate:
      oss << "hipEventCreate(";
      if (data->args.hipEventCreate.event == NULL) oss << "event=NULL";
      else oss << "event=" << data->args.hipEventCreate.event__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemGetAddressRange:
      oss << "hipMemGetAddressRange(";
      if (data->args.hipMemGetAddressRange.pbase == NULL) oss << "pbase=NULL";
      else oss << "pbase=" << data->args.hipMemGetAddressRange.pbase__val;
      if (data->args.hipMemGetAddressRange.psize == NULL) oss << ", psize=NULL";
      else oss << ", psize=" << data->args.hipMemGetAddressRange.psize__val;
      oss << ", dptr=" << data->args.hipMemGetAddressRange.dptr;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyFromSymbol:
      oss << "hipMemcpyFromSymbol(";
      oss << "dst=" << data->args.hipMemcpyFromSymbol.dst;
      oss << ", symbol=" << data->args.hipMemcpyFromSymbol.symbol;
      oss << ", sizeBytes=" << data->args.hipMemcpyFromSymbol.sizeBytes;
      oss << ", offset=" << data->args.hipMemcpyFromSymbol.offset;
      oss << ", kind=" << data->args.hipMemcpyFromSymbol.kind;
      oss << ")";
    break;
    case HIP_API_ID_hipArrayCreate:
      oss << "hipArrayCreate(";
      if (data->args.hipArrayCreate.pHandle == NULL) oss << "pHandle=NULL";
      else oss << "pHandle=" << (void*)data->args.hipArrayCreate.pHandle__val;
      if (data->args.hipArrayCreate.pAllocateArray == NULL) oss << ", pAllocateArray=NULL";
      else oss << ", pAllocateArray=" << data->args.hipArrayCreate.pAllocateArray__val;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamAttachMemAsync:
      oss << "hipStreamAttachMemAsync(";
      oss << "stream=" << data->args.hipStreamAttachMemAsync.stream;
      if (data->args.hipStreamAttachMemAsync.dev_ptr == NULL) oss << ", dev_ptr=NULL";
      else oss << ", dev_ptr=" << data->args.hipStreamAttachMemAsync.dev_ptr__val;
      oss << ", length=" << data->args.hipStreamAttachMemAsync.length;
      oss << ", flags=" << data->args.hipStreamAttachMemAsync.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamGetFlags:
      oss << "hipStreamGetFlags(";
      oss << "stream=" << data->args.hipStreamGetFlags.stream;
      if (data->args.hipStreamGetFlags.flags == NULL) oss << ", flags=NULL";
      else oss << ", flags=" << data->args.hipStreamGetFlags.flags__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMallocArray:
      oss << "hipMallocArray(";
      if (data->args.hipMallocArray.array == NULL) oss << "array=NULL";
      else oss << "array=" << (void*)data->args.hipMallocArray.array__val;
      if (data->args.hipMallocArray.desc == NULL) oss << ", desc=NULL";
      else oss << ", desc=" << data->args.hipMallocArray.desc__val;
      oss << ", width=" << data->args.hipMallocArray.width;
      oss << ", height=" << data->args.hipMallocArray.height;
      oss << ", flags=" << data->args.hipMallocArray.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxGetSharedMemConfig:
      oss << "hipCtxGetSharedMemConfig(";
      if (data->args.hipCtxGetSharedMemConfig.pConfig == NULL) oss << "pConfig=NULL";
      else oss << "pConfig=" << data->args.hipCtxGetSharedMemConfig.pConfig__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceDisablePeerAccess:
      oss << "hipDeviceDisablePeerAccess(";
      oss << "peerDeviceId=" << data->args.hipDeviceDisablePeerAccess.peerDeviceId;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize:
      oss << "hipModuleOccupancyMaxPotentialBlockSize(";
      if (data->args.hipModuleOccupancyMaxPotentialBlockSize.gridSize == NULL) oss << "gridSize=NULL";
      else oss << "gridSize=" << data->args.hipModuleOccupancyMaxPotentialBlockSize.gridSize__val;
      if (data->args.hipModuleOccupancyMaxPotentialBlockSize.blockSize == NULL) oss << ", blockSize=NULL";
      else oss << ", blockSize=" << data->args.hipModuleOccupancyMaxPotentialBlockSize.blockSize__val;
      oss << ", f=" << data->args.hipModuleOccupancyMaxPotentialBlockSize.f;
      oss << ", dynSharedMemPerBlk=" << data->args.hipModuleOccupancyMaxPotentialBlockSize.dynSharedMemPerBlk;
      oss << ", blockSizeLimit=" << data->args.hipModuleOccupancyMaxPotentialBlockSize.blockSizeLimit;
      oss << ")";
    break;
    case HIP_API_ID_hipMemPtrGetInfo:
      oss << "hipMemPtrGetInfo(";
      oss << "ptr=" << data->args.hipMemPtrGetInfo.ptr;
      if (data->args.hipMemPtrGetInfo.size == NULL) oss << ", size=NULL";
      else oss << ", size=" << data->args.hipMemPtrGetInfo.size__val;
      oss << ")";
    break;
    case HIP_API_ID_hipFuncGetAttribute:
      oss << "hipFuncGetAttribute(";
      if (data->args.hipFuncGetAttribute.value == NULL) oss << "value=NULL";
      else oss << "value=" << data->args.hipFuncGetAttribute.value__val;
      oss << ", attrib=" << data->args.hipFuncGetAttribute.attrib;
      oss << ", hfunc=" << data->args.hipFuncGetAttribute.hfunc;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxGetFlags:
      oss << "hipCtxGetFlags(";
      if (data->args.hipCtxGetFlags.flags == NULL) oss << "flags=NULL";
      else oss << "flags=" << data->args.hipCtxGetFlags.flags__val;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamDestroy:
      oss << "hipStreamDestroy(";
      oss << "stream=" << data->args.hipStreamDestroy.stream;
      oss << ")";
    break;
    case HIP_API_ID___hipPushCallConfiguration:
      oss << "__hipPushCallConfiguration(";
      oss << "gridDim=" << data->args.__hipPushCallConfiguration.gridDim;
      oss << ", blockDim=" << data->args.__hipPushCallConfiguration.blockDim;
      oss << ", sharedMem=" << data->args.__hipPushCallConfiguration.sharedMem;
      oss << ", stream=" << data->args.__hipPushCallConfiguration.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipMemset3DAsync:
      oss << "hipMemset3DAsync(";
      oss << "pitchedDevPtr=" << data->args.hipMemset3DAsync.pitchedDevPtr;
      oss << ", value=" << data->args.hipMemset3DAsync.value;
      oss << ", extent=" << data->args.hipMemset3DAsync.extent;
      oss << ", stream=" << data->args.hipMemset3DAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetPCIBusId:
      oss << "hipDeviceGetPCIBusId(";
      if (data->args.hipDeviceGetPCIBusId.pciBusId == NULL) oss << "pciBusId=NULL";
      else oss << "pciBusId=" << data->args.hipDeviceGetPCIBusId.pciBusId__val;
      oss << ", len=" << data->args.hipDeviceGetPCIBusId.len;
      oss << ", device=" << data->args.hipDeviceGetPCIBusId.device;
      oss << ")";
    break;
    case HIP_API_ID_hipInit:
      oss << "hipInit(";
      oss << "flags=" << data->args.hipInit.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyAtoH:
      oss << "hipMemcpyAtoH(";
      oss << "dst=" << data->args.hipMemcpyAtoH.dst;
      if (data->args.hipMemcpyAtoH.srcArray == NULL) oss << ", srcArray=NULL";
      else oss << ", srcArray=" << data->args.hipMemcpyAtoH.srcArray__val;
      oss << ", srcOffset=" << data->args.hipMemcpyAtoH.srcOffset;
      oss << ", count=" << data->args.hipMemcpyAtoH.count;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamGetPriority:
      oss << "hipStreamGetPriority(";
      oss << "stream=" << data->args.hipStreamGetPriority.stream;
      if (data->args.hipStreamGetPriority.priority == NULL) oss << ", priority=NULL";
      else oss << ", priority=" << data->args.hipStreamGetPriority.priority__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemset2D:
      oss << "hipMemset2D(";
      oss << "dst=" << data->args.hipMemset2D.dst;
      oss << ", pitch=" << data->args.hipMemset2D.pitch;
      oss << ", value=" << data->args.hipMemset2D.value;
      oss << ", width=" << data->args.hipMemset2D.width;
      oss << ", height=" << data->args.hipMemset2D.height;
      oss << ")";
    break;
    case HIP_API_ID_hipMemset2DAsync:
      oss << "hipMemset2DAsync(";
      oss << "dst=" << data->args.hipMemset2DAsync.dst;
      oss << ", pitch=" << data->args.hipMemset2DAsync.pitch;
      oss << ", value=" << data->args.hipMemset2DAsync.value;
      oss << ", width=" << data->args.hipMemset2DAsync.width;
      oss << ", height=" << data->args.hipMemset2DAsync.height;
      oss << ", stream=" << data->args.hipMemset2DAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceCanAccessPeer:
      oss << "hipDeviceCanAccessPeer(";
      if (data->args.hipDeviceCanAccessPeer.canAccessPeer == NULL) oss << "canAccessPeer=NULL";
      else oss << "canAccessPeer=" << data->args.hipDeviceCanAccessPeer.canAccessPeer__val;
      oss << ", deviceId=" << data->args.hipDeviceCanAccessPeer.deviceId;
      oss << ", peerDeviceId=" << data->args.hipDeviceCanAccessPeer.peerDeviceId;
      oss << ")";
    break;
    case HIP_API_ID_hipLaunchByPtr:
      oss << "hipLaunchByPtr(";
      oss << "hostFunction=" << data->args.hipLaunchByPtr.hostFunction;
      oss << ")";
    break;
    case HIP_API_ID_hipMemPrefetchAsync:
      oss << "hipMemPrefetchAsync(";
      oss << "dev_ptr=" << data->args.hipMemPrefetchAsync.dev_ptr;
      oss << ", count=" << data->args.hipMemPrefetchAsync.count;
      oss << ", device=" << data->args.hipMemPrefetchAsync.device;
      oss << ", stream=" << data->args.hipMemPrefetchAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxDestroy:
      oss << "hipCtxDestroy(";
      oss << "ctx=" << data->args.hipCtxDestroy.ctx;
      oss << ")";
    break;
    case HIP_API_ID_hipMemsetD16Async:
      oss << "hipMemsetD16Async(";
      oss << "dest=" << data->args.hipMemsetD16Async.dest;
      oss << ", value=" << data->args.hipMemsetD16Async.value;
      oss << ", count=" << data->args.hipMemsetD16Async.count;
      oss << ", stream=" << data->args.hipMemsetD16Async.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleUnload:
      oss << "hipModuleUnload(";
      oss << "module=" << data->args.hipModuleUnload.module;
      oss << ")";
    break;
    case HIP_API_ID_hipHostUnregister:
      oss << "hipHostUnregister(";
      oss << "hostPtr=" << data->args.hipHostUnregister.hostPtr;
      oss << ")";
    break;
    case HIP_API_ID_hipProfilerStop:
      oss << "hipProfilerStop(";
      oss << ")";
    break;
    case HIP_API_ID_hipExtStreamCreateWithCUMask:
      oss << "hipExtStreamCreateWithCUMask(";
      if (data->args.hipExtStreamCreateWithCUMask.stream == NULL) oss << "stream=NULL";
      else oss << "stream=" << data->args.hipExtStreamCreateWithCUMask.stream__val;
      oss << ", cuMaskSize=" << data->args.hipExtStreamCreateWithCUMask.cuMaskSize;
      if (data->args.hipExtStreamCreateWithCUMask.cuMask == NULL) oss << ", cuMask=NULL";
      else oss << ", cuMask=" << data->args.hipExtStreamCreateWithCUMask.cuMask__val;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamSynchronize:
      oss << "hipStreamSynchronize(";
      oss << "stream=" << data->args.hipStreamSynchronize.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipFreeHost:
      oss << "hipFreeHost(";
      oss << "ptr=" << data->args.hipFreeHost.ptr;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceSetCacheConfig:
      oss << "hipDeviceSetCacheConfig(";
      oss << "cacheConfig=" << data->args.hipDeviceSetCacheConfig.cacheConfig;
      oss << ")";
    break;
    case HIP_API_ID_hipGetErrorName:
      oss << "hipGetErrorName(";
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyHtoD:
      oss << "hipMemcpyHtoD(";
      oss << "dst=" << data->args.hipMemcpyHtoD.dst;
      oss << ", src=" << data->args.hipMemcpyHtoD.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyHtoD.sizeBytes;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleGetGlobal:
      oss << "hipModuleGetGlobal(";
      if (data->args.hipModuleGetGlobal.dptr == NULL) oss << "dptr=NULL";
      else oss << "dptr=" << data->args.hipModuleGetGlobal.dptr__val;
      if (data->args.hipModuleGetGlobal.bytes == NULL) oss << ", bytes=NULL";
      else oss << ", bytes=" << data->args.hipModuleGetGlobal.bytes__val;
      oss << ", hmod=" << data->args.hipModuleGetGlobal.hmod;
      if (data->args.hipModuleGetGlobal.name == NULL) oss << ", name=NULL";
      else oss << ", name=" << data->args.hipModuleGetGlobal.name__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyHtoA:
      oss << "hipMemcpyHtoA(";
      if (data->args.hipMemcpyHtoA.dstArray == NULL) oss << "dstArray=NULL";
      else oss << "dstArray=" << data->args.hipMemcpyHtoA.dstArray__val;
      oss << ", dstOffset=" << data->args.hipMemcpyHtoA.dstOffset;
      oss << ", srcHost=" << data->args.hipMemcpyHtoA.srcHost;
      oss << ", count=" << data->args.hipMemcpyHtoA.count;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxCreate:
      oss << "hipCtxCreate(";
      if (data->args.hipCtxCreate.ctx == NULL) oss << "ctx=NULL";
      else oss << "ctx=" << data->args.hipCtxCreate.ctx__val;
      oss << ", flags=" << data->args.hipCtxCreate.flags;
      oss << ", device=" << data->args.hipCtxCreate.device;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpy2D:
      oss << "hipMemcpy2D(";
      oss << "dst=" << data->args.hipMemcpy2D.dst;
      oss << ", dpitch=" << data->args.hipMemcpy2D.dpitch;
      oss << ", src=" << data->args.hipMemcpy2D.src;
      oss << ", spitch=" << data->args.hipMemcpy2D.spitch;
      oss << ", width=" << data->args.hipMemcpy2D.width;
      oss << ", height=" << data->args.hipMemcpy2D.height;
      oss << ", kind=" << data->args.hipMemcpy2D.kind;
      oss << ")";
    break;
    case HIP_API_ID_hipIpcCloseMemHandle:
      oss << "hipIpcCloseMemHandle(";
      oss << "devPtr=" << data->args.hipIpcCloseMemHandle.devPtr;
      oss << ")";
    break;
    case HIP_API_ID_hipChooseDevice:
      oss << "hipChooseDevice(";
      if (data->args.hipChooseDevice.device == NULL) oss << "device=NULL";
      else oss << "device=" << data->args.hipChooseDevice.device__val;
      if (data->args.hipChooseDevice.prop == NULL) oss << ", prop=NULL";
      else oss << ", prop=" << data->args.hipChooseDevice.prop__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceSetSharedMemConfig:
      oss << "hipDeviceSetSharedMemConfig(";
      oss << "config=" << data->args.hipDeviceSetSharedMemConfig.config;
      oss << ")";
    break;
    case HIP_API_ID_hipMallocMipmappedArray:
      oss << "hipMallocMipmappedArray(";
      if (data->args.hipMallocMipmappedArray.mipmappedArray == NULL) oss << "mipmappedArray=NULL";
      else oss << "mipmappedArray=" << data->args.hipMallocMipmappedArray.mipmappedArray__val;
      if (data->args.hipMallocMipmappedArray.desc == NULL) oss << ", desc=NULL";
      else oss << ", desc=" << data->args.hipMallocMipmappedArray.desc__val;
      oss << ", extent=" << data->args.hipMallocMipmappedArray.extent;
      oss << ", numLevels=" << data->args.hipMallocMipmappedArray.numLevels;
      oss << ", flags=" << data->args.hipMallocMipmappedArray.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipSetupArgument:
      oss << "hipSetupArgument(";
      oss << "arg=" << data->args.hipSetupArgument.arg;
      oss << ", size=" << data->args.hipSetupArgument.size;
      oss << ", offset=" << data->args.hipSetupArgument.offset;
      oss << ")";
    break;
    case HIP_API_ID_hipIpcGetEventHandle:
      oss << "hipIpcGetEventHandle(";
      if (data->args.hipIpcGetEventHandle.handle == NULL) oss << "handle=NULL";
      else oss << "handle=" << data->args.hipIpcGetEventHandle.handle__val;
      oss << ", event=" << data->args.hipIpcGetEventHandle.event;
      oss << ")";
    break;
    case HIP_API_ID_hipFreeArray:
      oss << "hipFreeArray(";
      if (data->args.hipFreeArray.array == NULL) oss << "array=NULL";
      else oss << "array=" << data->args.hipFreeArray.array__val;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxSetCacheConfig:
      oss << "hipCtxSetCacheConfig(";
      oss << "cacheConfig=" << data->args.hipCtxSetCacheConfig.cacheConfig;
      oss << ")";
    break;
    case HIP_API_ID_hipFuncSetCacheConfig:
      oss << "hipFuncSetCacheConfig(";
      oss << "func=" << data->args.hipFuncSetCacheConfig.func;
      oss << ", config=" << data->args.hipFuncSetCacheConfig.config;
      oss << ")";
    break;
    case HIP_API_ID_hipLaunchKernel:
      oss << "hipLaunchKernel(";
      oss << "function_address=" << data->args.hipLaunchKernel.function_address;
      oss << ", numBlocks=" << data->args.hipLaunchKernel.numBlocks;
      oss << ", dimBlocks=" << data->args.hipLaunchKernel.dimBlocks;
      if (data->args.hipLaunchKernel.args == NULL) oss << ", args=NULL";
      else oss << ", args=" << data->args.hipLaunchKernel.args__val;
      oss << ", sharedMemBytes=" << data->args.hipLaunchKernel.sharedMemBytes;
      oss << ", stream=" << data->args.hipLaunchKernel.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:
      oss << "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(";
      if (data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks == NULL) oss << "numBlocks=NULL";
      else oss << "numBlocks=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks__val;
      oss << ", f=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.f;
      oss << ", blockSize=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.blockSize;
      oss << ", dynSharedMemPerBlk=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.dynSharedMemPerBlk;
      oss << ", flags=" << data->args.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleGetTexRef:
      oss << "hipModuleGetTexRef(";
      if (data->args.hipModuleGetTexRef.texRef == NULL) oss << "texRef=NULL";
      else oss << "texRef=" << (void*)data->args.hipModuleGetTexRef.texRef__val;
      oss << ", hmod=" << data->args.hipModuleGetTexRef.hmod;
      if (data->args.hipModuleGetTexRef.name == NULL) oss << ", name=NULL";
      else oss << ", name=" << data->args.hipModuleGetTexRef.name__val;
      oss << ")";
    break;
    case HIP_API_ID_hipFuncSetAttribute:
      oss << "hipFuncSetAttribute(";
      oss << "func=" << data->args.hipFuncSetAttribute.func;
      oss << ", attr=" << data->args.hipFuncSetAttribute.attr;
      oss << ", value=" << data->args.hipFuncSetAttribute.value;
      oss << ")";
    break;
    case HIP_API_ID_hipEventElapsedTime:
      oss << "hipEventElapsedTime(";
      if (data->args.hipEventElapsedTime.ms == NULL) oss << "ms=NULL";
      else oss << "ms=" << data->args.hipEventElapsedTime.ms__val;
      oss << ", start=" << data->args.hipEventElapsedTime.start;
      oss << ", stop=" << data->args.hipEventElapsedTime.stop;
      oss << ")";
    break;
    case HIP_API_ID_hipConfigureCall:
      oss << "hipConfigureCall(";
      oss << "gridDim=" << data->args.hipConfigureCall.gridDim;
      oss << ", blockDim=" << data->args.hipConfigureCall.blockDim;
      oss << ", sharedMem=" << data->args.hipConfigureCall.sharedMem;
      oss << ", stream=" << data->args.hipConfigureCall.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipMemAdvise:
      oss << "hipMemAdvise(";
      oss << "dev_ptr=" << data->args.hipMemAdvise.dev_ptr;
      oss << ", count=" << data->args.hipMemAdvise.count;
      oss << ", advice=" << data->args.hipMemAdvise.advice;
      oss << ", device=" << data->args.hipMemAdvise.device;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpy3DAsync:
      oss << "hipMemcpy3DAsync(";
      if (data->args.hipMemcpy3DAsync.p == NULL) oss << "p=NULL";
      else oss << "p=" << data->args.hipMemcpy3DAsync.p__val;
      oss << ", stream=" << data->args.hipMemcpy3DAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipEventDestroy:
      oss << "hipEventDestroy(";
      oss << "event=" << data->args.hipEventDestroy.event;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxPopCurrent:
      oss << "hipCtxPopCurrent(";
      if (data->args.hipCtxPopCurrent.ctx == NULL) oss << "ctx=NULL";
      else oss << "ctx=" << data->args.hipCtxPopCurrent.ctx__val;
      oss << ")";
    break;
    case HIP_API_ID_hipGetSymbolAddress:
      oss << "hipGetSymbolAddress(";
      if (data->args.hipGetSymbolAddress.devPtr == NULL) oss << "devPtr=NULL";
      else oss << "devPtr=" << data->args.hipGetSymbolAddress.devPtr__val;
      oss << ", symbol=" << data->args.hipGetSymbolAddress.symbol;
      oss << ")";
    break;
    case HIP_API_ID_hipHostGetFlags:
      oss << "hipHostGetFlags(";
      if (data->args.hipHostGetFlags.flagsPtr == NULL) oss << "flagsPtr=NULL";
      else oss << "flagsPtr=" << data->args.hipHostGetFlags.flagsPtr__val;
      oss << ", hostPtr=" << data->args.hipHostGetFlags.hostPtr;
      oss << ")";
    break;
    case HIP_API_ID_hipHostMalloc:
      oss << "hipHostMalloc(";
      if (data->args.hipHostMalloc.ptr == NULL) oss << "ptr=NULL";
      else oss << "ptr=" << data->args.hipHostMalloc.ptr__val;
      oss << ", size=" << data->args.hipHostMalloc.size;
      oss << ", flags=" << data->args.hipHostMalloc.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxSetSharedMemConfig:
      oss << "hipCtxSetSharedMemConfig(";
      oss << "config=" << data->args.hipCtxSetSharedMemConfig.config;
      oss << ")";
    break;
    case HIP_API_ID_hipFreeMipmappedArray:
      oss << "hipFreeMipmappedArray(";
      oss << "mipmappedArray=" << data->args.hipFreeMipmappedArray.mipmappedArray;
      oss << ")";
    break;
    case HIP_API_ID_hipMemGetInfo:
      oss << "hipMemGetInfo(";
      if (data->args.hipMemGetInfo.free == NULL) oss << "free=NULL";
      else oss << "free=" << data->args.hipMemGetInfo.free__val;
      if (data->args.hipMemGetInfo.total == NULL) oss << ", total=NULL";
      else oss << ", total=" << data->args.hipMemGetInfo.total__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceReset:
      oss << "hipDeviceReset(";
      oss << ")";
    break;
    case HIP_API_ID_hipMemset:
      oss << "hipMemset(";
      oss << "dst=" << data->args.hipMemset.dst;
      oss << ", value=" << data->args.hipMemset.value;
      oss << ", sizeBytes=" << data->args.hipMemset.sizeBytes;
      oss << ")";
    break;
    case HIP_API_ID_hipMemsetD8:
      oss << "hipMemsetD8(";
      oss << "dest=" << data->args.hipMemsetD8.dest;
      oss << ", value=" << data->args.hipMemsetD8.value;
      oss << ", count=" << data->args.hipMemsetD8.count;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyParam2DAsync:
      oss << "hipMemcpyParam2DAsync(";
      if (data->args.hipMemcpyParam2DAsync.pCopy == NULL) oss << "pCopy=NULL";
      else oss << "pCopy=" << data->args.hipMemcpyParam2DAsync.pCopy__val;
      oss << ", stream=" << data->args.hipMemcpyParam2DAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipHostRegister:
      oss << "hipHostRegister(";
      oss << "hostPtr=" << data->args.hipHostRegister.hostPtr;
      oss << ", sizeBytes=" << data->args.hipHostRegister.sizeBytes;
      oss << ", flags=" << data->args.hipHostRegister.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipDriverGetVersion:
      oss << "hipDriverGetVersion(";
      if (data->args.hipDriverGetVersion.driverVersion == NULL) oss << "driverVersion=NULL";
      else oss << "driverVersion=" << data->args.hipDriverGetVersion.driverVersion__val;
      oss << ")";
    break;
    case HIP_API_ID_hipArray3DCreate:
      oss << "hipArray3DCreate(";
      if (data->args.hipArray3DCreate.array == NULL) oss << "array=NULL";
      else oss << "array=" << (void*)data->args.hipArray3DCreate.array__val;
      if (data->args.hipArray3DCreate.pAllocateArray == NULL) oss << ", pAllocateArray=NULL";
      else oss << ", pAllocateArray=" << data->args.hipArray3DCreate.pAllocateArray__val;
      oss << ")";
    break;
    case HIP_API_ID_hipIpcOpenMemHandle:
      oss << "hipIpcOpenMemHandle(";
      if (data->args.hipIpcOpenMemHandle.devPtr == NULL) oss << "devPtr=NULL";
      else oss << "devPtr=" << data->args.hipIpcOpenMemHandle.devPtr__val;
      oss << ", handle=" << data->args.hipIpcOpenMemHandle.handle;
      oss << ", flags=" << data->args.hipIpcOpenMemHandle.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipGetLastError:
      oss << "hipGetLastError(";
      oss << ")";
    break;
    case HIP_API_ID_hipGetDeviceFlags:
      oss << "hipGetDeviceFlags(";
      if (data->args.hipGetDeviceFlags.flags == NULL) oss << "flags=NULL";
      else oss << "flags=" << data->args.hipGetDeviceFlags.flags__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetSharedMemConfig:
      oss << "hipDeviceGetSharedMemConfig(";
      if (data->args.hipDeviceGetSharedMemConfig.pConfig == NULL) oss << "pConfig=NULL";
      else oss << "pConfig=" << data->args.hipDeviceGetSharedMemConfig.pConfig__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDrvMemcpy3D:
      oss << "hipDrvMemcpy3D(";
      if (data->args.hipDrvMemcpy3D.pCopy == NULL) oss << "pCopy=NULL";
      else oss << "pCopy=" << data->args.hipDrvMemcpy3D.pCopy__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpy2DFromArray:
      oss << "hipMemcpy2DFromArray(";
      oss << "dst=" << data->args.hipMemcpy2DFromArray.dst;
      oss << ", dpitch=" << data->args.hipMemcpy2DFromArray.dpitch;
      oss << ", src=" << data->args.hipMemcpy2DFromArray.src;
      oss << ", wOffset=" << data->args.hipMemcpy2DFromArray.wOffset;
      oss << ", hOffset=" << data->args.hipMemcpy2DFromArray.hOffset;
      oss << ", width=" << data->args.hipMemcpy2DFromArray.width;
      oss << ", height=" << data->args.hipMemcpy2DFromArray.height;
      oss << ", kind=" << data->args.hipMemcpy2DFromArray.kind;
      oss << ")";
    break;
    case HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:
      oss << "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(";
      if (data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks == NULL) oss << "numBlocks=NULL";
      else oss << "numBlocks=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks__val;
      oss << ", f=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.f;
      oss << ", blockSize=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.blockSize;
      oss << ", dynamicSMemSize=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.dynamicSMemSize;
      oss << ", flags=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipSetDeviceFlags:
      oss << "hipSetDeviceFlags(";
      oss << "flags=" << data->args.hipSetDeviceFlags.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipHccModuleLaunchKernel:
      oss << "hipHccModuleLaunchKernel(";
      oss << "f=" << data->args.hipHccModuleLaunchKernel.f;
      oss << ", globalWorkSizeX=" << data->args.hipHccModuleLaunchKernel.globalWorkSizeX;
      oss << ", globalWorkSizeY=" << data->args.hipHccModuleLaunchKernel.globalWorkSizeY;
      oss << ", globalWorkSizeZ=" << data->args.hipHccModuleLaunchKernel.globalWorkSizeZ;
      oss << ", blockDimX=" << data->args.hipHccModuleLaunchKernel.blockDimX;
      oss << ", blockDimY=" << data->args.hipHccModuleLaunchKernel.blockDimY;
      oss << ", blockDimZ=" << data->args.hipHccModuleLaunchKernel.blockDimZ;
      oss << ", sharedMemBytes=" << data->args.hipHccModuleLaunchKernel.sharedMemBytes;
      oss << ", hStream=" << data->args.hipHccModuleLaunchKernel.hStream;
      if (data->args.hipHccModuleLaunchKernel.kernelParams == NULL) oss << ", kernelParams=NULL";
      else oss << ", kernelParams=" << data->args.hipHccModuleLaunchKernel.kernelParams__val;
      if (data->args.hipHccModuleLaunchKernel.extra == NULL) oss << ", extra=NULL";
      else oss << ", extra=" << data->args.hipHccModuleLaunchKernel.extra__val;
      oss << ", startEvent=" << data->args.hipHccModuleLaunchKernel.startEvent;
      oss << ", stopEvent=" << data->args.hipHccModuleLaunchKernel.stopEvent;
      oss << ")";
    break;
    case HIP_API_ID_hipFree:
      oss << "hipFree(";
      oss << "ptr=" << data->args.hipFree.ptr;
      oss << ")";
    break;
    case HIP_API_ID_hipOccupancyMaxPotentialBlockSize:
      oss << "hipOccupancyMaxPotentialBlockSize(";
      if (data->args.hipOccupancyMaxPotentialBlockSize.gridSize == NULL) oss << "gridSize=NULL";
      else oss << "gridSize=" << data->args.hipOccupancyMaxPotentialBlockSize.gridSize__val;
      if (data->args.hipOccupancyMaxPotentialBlockSize.blockSize == NULL) oss << ", blockSize=NULL";
      else oss << ", blockSize=" << data->args.hipOccupancyMaxPotentialBlockSize.blockSize__val;
      oss << ", f=" << data->args.hipOccupancyMaxPotentialBlockSize.f;
      oss << ", dynSharedMemPerBlk=" << data->args.hipOccupancyMaxPotentialBlockSize.dynSharedMemPerBlk;
      oss << ", blockSizeLimit=" << data->args.hipOccupancyMaxPotentialBlockSize.blockSizeLimit;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetAttribute:
      oss << "hipDeviceGetAttribute(";
      if (data->args.hipDeviceGetAttribute.pi == NULL) oss << "pi=NULL";
      else oss << "pi=" << data->args.hipDeviceGetAttribute.pi__val;
      oss << ", attr=" << data->args.hipDeviceGetAttribute.attr;
      oss << ", deviceId=" << data->args.hipDeviceGetAttribute.deviceId;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceComputeCapability:
      oss << "hipDeviceComputeCapability(";
      if (data->args.hipDeviceComputeCapability.major == NULL) oss << "major=NULL";
      else oss << "major=" << data->args.hipDeviceComputeCapability.major__val;
      if (data->args.hipDeviceComputeCapability.minor == NULL) oss << ", minor=NULL";
      else oss << ", minor=" << data->args.hipDeviceComputeCapability.minor__val;
      oss << ", device=" << data->args.hipDeviceComputeCapability.device;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxDisablePeerAccess:
      oss << "hipCtxDisablePeerAccess(";
      oss << "peerCtx=" << data->args.hipCtxDisablePeerAccess.peerCtx;
      oss << ")";
    break;
    case HIP_API_ID_hipMallocManaged:
      oss << "hipMallocManaged(";
      if (data->args.hipMallocManaged.dev_ptr == NULL) oss << "dev_ptr=NULL";
      else oss << "dev_ptr=" << data->args.hipMallocManaged.dev_ptr__val;
      oss << ", size=" << data->args.hipMallocManaged.size;
      oss << ", flags=" << data->args.hipMallocManaged.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetByPCIBusId:
      oss << "hipDeviceGetByPCIBusId(";
      if (data->args.hipDeviceGetByPCIBusId.device == NULL) oss << "device=NULL";
      else oss << "device=" << data->args.hipDeviceGetByPCIBusId.device__val;
      if (data->args.hipDeviceGetByPCIBusId.pciBusId == NULL) oss << ", pciBusId=NULL";
      else oss << ", pciBusId=" << data->args.hipDeviceGetByPCIBusId.pciBusId__val;
      oss << ")";
    break;
    case HIP_API_ID_hipIpcGetMemHandle:
      oss << "hipIpcGetMemHandle(";
      if (data->args.hipIpcGetMemHandle.handle == NULL) oss << "handle=NULL";
      else oss << "handle=" << data->args.hipIpcGetMemHandle.handle__val;
      oss << ", devPtr=" << data->args.hipIpcGetMemHandle.devPtr;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyHtoDAsync:
      oss << "hipMemcpyHtoDAsync(";
      oss << "dst=" << data->args.hipMemcpyHtoDAsync.dst;
      oss << ", src=" << data->args.hipMemcpyHtoDAsync.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyHtoDAsync.sizeBytes;
      oss << ", stream=" << data->args.hipMemcpyHtoDAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxGetDevice:
      oss << "hipCtxGetDevice(";
      if (data->args.hipCtxGetDevice.device == NULL) oss << "device=NULL";
      else oss << "device=" << data->args.hipCtxGetDevice.device__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoD:
      oss << "hipMemcpyDtoD(";
      oss << "dst=" << data->args.hipMemcpyDtoD.dst;
      oss << ", src=" << data->args.hipMemcpyDtoD.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyDtoD.sizeBytes;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleLoadData:
      oss << "hipModuleLoadData(";
      if (data->args.hipModuleLoadData.module == NULL) oss << "module=NULL";
      else oss << "module=" << data->args.hipModuleLoadData.module__val;
      oss << ", image=" << data->args.hipModuleLoadData.image;
      oss << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxRelease:
      oss << "hipDevicePrimaryCtxRelease(";
      oss << "dev=" << data->args.hipDevicePrimaryCtxRelease.dev;
      oss << ")";
    break;
    case HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor:
      oss << "hipOccupancyMaxActiveBlocksPerMultiprocessor(";
      if (data->args.hipOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks == NULL) oss << "numBlocks=NULL";
      else oss << "numBlocks=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks__val;
      oss << ", f=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessor.f;
      oss << ", blockSize=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessor.blockSize;
      oss << ", dynamicSMemSize=" << data->args.hipOccupancyMaxActiveBlocksPerMultiprocessor.dynamicSMemSize;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxSetCurrent:
      oss << "hipCtxSetCurrent(";
      oss << "ctx=" << data->args.hipCtxSetCurrent.ctx;
      oss << ")";
    break;
    case HIP_API_ID_hipGetErrorString:
      oss << "hipGetErrorString(";
      oss << ")";
    break;
    case HIP_API_ID_hipStreamCreate:
      oss << "hipStreamCreate(";
      if (data->args.hipStreamCreate.stream == NULL) oss << "stream=NULL";
      else oss << "stream=" << data->args.hipStreamCreate.stream__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxRetain:
      oss << "hipDevicePrimaryCtxRetain(";
      if (data->args.hipDevicePrimaryCtxRetain.pctx == NULL) oss << "pctx=NULL";
      else oss << "pctx=" << data->args.hipDevicePrimaryCtxRetain.pctx__val;
      oss << ", dev=" << data->args.hipDevicePrimaryCtxRetain.dev;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGet:
      oss << "hipDeviceGet(";
      if (data->args.hipDeviceGet.device == NULL) oss << "device=NULL";
      else oss << "device=" << data->args.hipDeviceGet.device__val;
      oss << ", ordinal=" << data->args.hipDeviceGet.ordinal;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamCreateWithFlags:
      oss << "hipStreamCreateWithFlags(";
      if (data->args.hipStreamCreateWithFlags.stream == NULL) oss << "stream=NULL";
      else oss << "stream=" << data->args.hipStreamCreateWithFlags.stream__val;
      oss << ", flags=" << data->args.hipStreamCreateWithFlags.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyFromArray:
      oss << "hipMemcpyFromArray(";
      oss << "dst=" << data->args.hipMemcpyFromArray.dst;
      oss << ", srcArray=" << data->args.hipMemcpyFromArray.srcArray;
      oss << ", wOffset=" << data->args.hipMemcpyFromArray.wOffset;
      oss << ", hOffset=" << data->args.hipMemcpyFromArray.hOffset;
      oss << ", count=" << data->args.hipMemcpyFromArray.count;
      oss << ", kind=" << data->args.hipMemcpyFromArray.kind;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpy2DAsync:
      oss << "hipMemcpy2DAsync(";
      oss << "dst=" << data->args.hipMemcpy2DAsync.dst;
      oss << ", dpitch=" << data->args.hipMemcpy2DAsync.dpitch;
      oss << ", src=" << data->args.hipMemcpy2DAsync.src;
      oss << ", spitch=" << data->args.hipMemcpy2DAsync.spitch;
      oss << ", width=" << data->args.hipMemcpy2DAsync.width;
      oss << ", height=" << data->args.hipMemcpy2DAsync.height;
      oss << ", kind=" << data->args.hipMemcpy2DAsync.kind;
      oss << ", stream=" << data->args.hipMemcpy2DAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipFuncGetAttributes:
      oss << "hipFuncGetAttributes(";
      if (data->args.hipFuncGetAttributes.attr == NULL) oss << "attr=NULL";
      else oss << "attr=" << data->args.hipFuncGetAttributes.attr__val;
      oss << ", func=" << data->args.hipFuncGetAttributes.func;
      oss << ")";
    break;
    case HIP_API_ID_hipGetSymbolSize:
      oss << "hipGetSymbolSize(";
      if (data->args.hipGetSymbolSize.size == NULL) oss << "size=NULL";
      else oss << "size=" << data->args.hipGetSymbolSize.size__val;
      oss << ", symbol=" << data->args.hipGetSymbolSize.symbol;
      oss << ")";
    break;
    case HIP_API_ID_hipHostFree:
      oss << "hipHostFree(";
      oss << "ptr=" << data->args.hipHostFree.ptr;
      oss << ")";
    break;
    case HIP_API_ID_hipEventCreateWithFlags:
      oss << "hipEventCreateWithFlags(";
      if (data->args.hipEventCreateWithFlags.event == NULL) oss << "event=NULL";
      else oss << "event=" << data->args.hipEventCreateWithFlags.event__val;
      oss << ", flags=" << data->args.hipEventCreateWithFlags.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamQuery:
      oss << "hipStreamQuery(";
      oss << "stream=" << data->args.hipStreamQuery.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpy3D:
      oss << "hipMemcpy3D(";
      if (data->args.hipMemcpy3D.p == NULL) oss << "p=NULL";
      else oss << "p=" << data->args.hipMemcpy3D.p__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyToSymbol:
      oss << "hipMemcpyToSymbol(";
      oss << "symbol=" << data->args.hipMemcpyToSymbol.symbol;
      oss << ", src=" << data->args.hipMemcpyToSymbol.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyToSymbol.sizeBytes;
      oss << ", offset=" << data->args.hipMemcpyToSymbol.offset;
      oss << ", kind=" << data->args.hipMemcpyToSymbol.kind;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpy:
      oss << "hipMemcpy(";
      oss << "dst=" << data->args.hipMemcpy.dst;
      oss << ", src=" << data->args.hipMemcpy.src;
      oss << ", sizeBytes=" << data->args.hipMemcpy.sizeBytes;
      oss << ", kind=" << data->args.hipMemcpy.kind;
      oss << ")";
    break;
    case HIP_API_ID_hipPeekAtLastError:
      oss << "hipPeekAtLastError(";
      oss << ")";
    break;
    case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
      oss << "hipExtLaunchMultiKernelMultiDevice(";
      if (data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList == NULL) oss << "launchParamsList=NULL";
      else oss << "launchParamsList=" << data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList__val;
      oss << ", numDevices=" << data->args.hipExtLaunchMultiKernelMultiDevice.numDevices;
      oss << ", flags=" << data->args.hipExtLaunchMultiKernelMultiDevice.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipHostAlloc:
      oss << "hipHostAlloc(";
      if (data->args.hipHostAlloc.ptr == NULL) oss << "ptr=NULL";
      else oss << "ptr=" << data->args.hipHostAlloc.ptr__val;
      oss << ", size=" << data->args.hipHostAlloc.size;
      oss << ", flags=" << data->args.hipHostAlloc.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipStreamAddCallback:
      oss << "hipStreamAddCallback(";
      oss << "stream=" << data->args.hipStreamAddCallback.stream;
      oss << ", callback=" << data->args.hipStreamAddCallback.callback;
      oss << ", userData=" << data->args.hipStreamAddCallback.userData;
      oss << ", flags=" << data->args.hipStreamAddCallback.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyToArray:
      oss << "hipMemcpyToArray(";
      if (data->args.hipMemcpyToArray.dst == NULL) oss << "dst=NULL";
      else oss << "dst=" << data->args.hipMemcpyToArray.dst__val;
      oss << ", wOffset=" << data->args.hipMemcpyToArray.wOffset;
      oss << ", hOffset=" << data->args.hipMemcpyToArray.hOffset;
      oss << ", src=" << data->args.hipMemcpyToArray.src;
      oss << ", count=" << data->args.hipMemcpyToArray.count;
      oss << ", kind=" << data->args.hipMemcpyToArray.kind;
      oss << ")";
    break;
    case HIP_API_ID_hipMemsetD32:
      oss << "hipMemsetD32(";
      oss << "dest=" << data->args.hipMemsetD32.dest;
      oss << ", value=" << data->args.hipMemsetD32.value;
      oss << ", count=" << data->args.hipMemsetD32.count;
      oss << ")";
    break;
    case HIP_API_ID_hipExtModuleLaunchKernel:
      oss << "hipExtModuleLaunchKernel(";
      oss << "f=" << data->args.hipExtModuleLaunchKernel.f;
      oss << ", globalWorkSizeX=" << data->args.hipExtModuleLaunchKernel.globalWorkSizeX;
      oss << ", globalWorkSizeY=" << data->args.hipExtModuleLaunchKernel.globalWorkSizeY;
      oss << ", globalWorkSizeZ=" << data->args.hipExtModuleLaunchKernel.globalWorkSizeZ;
      oss << ", localWorkSizeX=" << data->args.hipExtModuleLaunchKernel.localWorkSizeX;
      oss << ", localWorkSizeY=" << data->args.hipExtModuleLaunchKernel.localWorkSizeY;
      oss << ", localWorkSizeZ=" << data->args.hipExtModuleLaunchKernel.localWorkSizeZ;
      oss << ", sharedMemBytes=" << data->args.hipExtModuleLaunchKernel.sharedMemBytes;
      oss << ", hStream=" << data->args.hipExtModuleLaunchKernel.hStream;
      if (data->args.hipExtModuleLaunchKernel.kernelParams == NULL) oss << ", kernelParams=NULL";
      else oss << ", kernelParams=" << data->args.hipExtModuleLaunchKernel.kernelParams__val;
      if (data->args.hipExtModuleLaunchKernel.extra == NULL) oss << ", extra=NULL";
      else oss << ", extra=" << data->args.hipExtModuleLaunchKernel.extra__val;
      oss << ", startEvent=" << data->args.hipExtModuleLaunchKernel.startEvent;
      oss << ", stopEvent=" << data->args.hipExtModuleLaunchKernel.stopEvent;
      oss << ", flags=" << data->args.hipExtModuleLaunchKernel.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceSynchronize:
      oss << "hipDeviceSynchronize(";
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetCacheConfig:
      oss << "hipDeviceGetCacheConfig(";
      if (data->args.hipDeviceGetCacheConfig.cacheConfig == NULL) oss << "cacheConfig=NULL";
      else oss << "cacheConfig=" << data->args.hipDeviceGetCacheConfig.cacheConfig__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMalloc3D:
      oss << "hipMalloc3D(";
      if (data->args.hipMalloc3D.pitchedDevPtr == NULL) oss << "pitchedDevPtr=NULL";
      else oss << "pitchedDevPtr=" << data->args.hipMalloc3D.pitchedDevPtr__val;
      oss << ", extent=" << data->args.hipMalloc3D.extent;
      oss << ")";
    break;
    case HIP_API_ID_hipPointerGetAttributes:
      oss << "hipPointerGetAttributes(";
      if (data->args.hipPointerGetAttributes.attributes == NULL) oss << "attributes=NULL";
      else oss << "attributes=" << data->args.hipPointerGetAttributes.attributes__val;
      oss << ", ptr=" << data->args.hipPointerGetAttributes.ptr;
      oss << ")";
    break;
    case HIP_API_ID_hipMemsetAsync:
      oss << "hipMemsetAsync(";
      oss << "dst=" << data->args.hipMemsetAsync.dst;
      oss << ", value=" << data->args.hipMemsetAsync.value;
      oss << ", sizeBytes=" << data->args.hipMemsetAsync.sizeBytes;
      oss << ", stream=" << data->args.hipMemsetAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetName:
      oss << "hipDeviceGetName(";
      if (data->args.hipDeviceGetName.name == NULL) oss << "name=NULL";
      else oss << "name=" << data->args.hipDeviceGetName.name__val;
      oss << ", len=" << data->args.hipDeviceGetName.len;
      oss << ", device=" << data->args.hipDeviceGetName.device;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags:
      oss << "hipModuleOccupancyMaxPotentialBlockSizeWithFlags(";
      if (data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize == NULL) oss << "gridSize=NULL";
      else oss << "gridSize=" << data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.gridSize__val;
      if (data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize == NULL) oss << ", blockSize=NULL";
      else oss << ", blockSize=" << data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSize__val;
      oss << ", f=" << data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.f;
      oss << ", dynSharedMemPerBlk=" << data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.dynSharedMemPerBlk;
      oss << ", blockSizeLimit=" << data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.blockSizeLimit;
      oss << ", flags=" << data->args.hipModuleOccupancyMaxPotentialBlockSizeWithFlags.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxPushCurrent:
      oss << "hipCtxPushCurrent(";
      oss << "ctx=" << data->args.hipCtxPushCurrent.ctx;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyPeer:
      oss << "hipMemcpyPeer(";
      oss << "dst=" << data->args.hipMemcpyPeer.dst;
      oss << ", dstDeviceId=" << data->args.hipMemcpyPeer.dstDeviceId;
      oss << ", src=" << data->args.hipMemcpyPeer.src;
      oss << ", srcDeviceId=" << data->args.hipMemcpyPeer.srcDeviceId;
      oss << ", sizeBytes=" << data->args.hipMemcpyPeer.sizeBytes;
      oss << ")";
    break;
    case HIP_API_ID_hipEventSynchronize:
      oss << "hipEventSynchronize(";
      oss << "event=" << data->args.hipEventSynchronize.event;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoDAsync:
      oss << "hipMemcpyDtoDAsync(";
      oss << "dst=" << data->args.hipMemcpyDtoDAsync.dst;
      oss << ", src=" << data->args.hipMemcpyDtoDAsync.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyDtoDAsync.sizeBytes;
      oss << ", stream=" << data->args.hipMemcpyDtoDAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipProfilerStart:
      oss << "hipProfilerStart(";
      oss << ")";
    break;
    case HIP_API_ID_hipExtMallocWithFlags:
      oss << "hipExtMallocWithFlags(";
      if (data->args.hipExtMallocWithFlags.ptr == NULL) oss << "ptr=NULL";
      else oss << "ptr=" << data->args.hipExtMallocWithFlags.ptr__val;
      oss << ", sizeBytes=" << data->args.hipExtMallocWithFlags.sizeBytes;
      oss << ", flags=" << data->args.hipExtMallocWithFlags.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipCtxEnablePeerAccess:
      oss << "hipCtxEnablePeerAccess(";
      oss << "peerCtx=" << data->args.hipCtxEnablePeerAccess.peerCtx;
      oss << ", flags=" << data->args.hipCtxEnablePeerAccess.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipMemAllocHost:
      oss << "hipMemAllocHost(";
      if (data->args.hipMemAllocHost.ptr == NULL) oss << "ptr=NULL";
      else oss << "ptr=" << data->args.hipMemAllocHost.ptr__val;
      oss << ", size=" << data->args.hipMemAllocHost.size;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoHAsync:
      oss << "hipMemcpyDtoHAsync(";
      oss << "dst=" << data->args.hipMemcpyDtoHAsync.dst;
      oss << ", src=" << data->args.hipMemcpyDtoHAsync.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyDtoHAsync.sizeBytes;
      oss << ", stream=" << data->args.hipMemcpyDtoHAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleLaunchKernel:
      oss << "hipModuleLaunchKernel(";
      oss << "f=" << data->args.hipModuleLaunchKernel.f;
      oss << ", gridDimX=" << data->args.hipModuleLaunchKernel.gridDimX;
      oss << ", gridDimY=" << data->args.hipModuleLaunchKernel.gridDimY;
      oss << ", gridDimZ=" << data->args.hipModuleLaunchKernel.gridDimZ;
      oss << ", blockDimX=" << data->args.hipModuleLaunchKernel.blockDimX;
      oss << ", blockDimY=" << data->args.hipModuleLaunchKernel.blockDimY;
      oss << ", blockDimZ=" << data->args.hipModuleLaunchKernel.blockDimZ;
      oss << ", sharedMemBytes=" << data->args.hipModuleLaunchKernel.sharedMemBytes;
      oss << ", stream=" << data->args.hipModuleLaunchKernel.stream;
      if (data->args.hipModuleLaunchKernel.kernelParams == NULL) oss << ", kernelParams=NULL";
      else oss << ", kernelParams=" << data->args.hipModuleLaunchKernel.kernelParams__val;
      if (data->args.hipModuleLaunchKernel.extra == NULL) oss << ", extra=NULL";
      else oss << ", extra=" << data->args.hipModuleLaunchKernel.extra__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemAllocPitch:
      oss << "hipMemAllocPitch(";
      if (data->args.hipMemAllocPitch.dptr == NULL) oss << "dptr=NULL";
      else oss << "dptr=" << data->args.hipMemAllocPitch.dptr__val;
      if (data->args.hipMemAllocPitch.pitch == NULL) oss << ", pitch=NULL";
      else oss << ", pitch=" << data->args.hipMemAllocPitch.pitch__val;
      oss << ", widthInBytes=" << data->args.hipMemAllocPitch.widthInBytes;
      oss << ", height=" << data->args.hipMemAllocPitch.height;
      oss << ", elementSizeBytes=" << data->args.hipMemAllocPitch.elementSizeBytes;
      oss << ")";
    break;
    case HIP_API_ID_hipExtLaunchKernel:
      oss << "hipExtLaunchKernel(";
      oss << "function_address=" << data->args.hipExtLaunchKernel.function_address;
      oss << ", numBlocks=" << data->args.hipExtLaunchKernel.numBlocks;
      oss << ", dimBlocks=" << data->args.hipExtLaunchKernel.dimBlocks;
      if (data->args.hipExtLaunchKernel.args == NULL) oss << ", args=NULL";
      else oss << ", args=" << data->args.hipExtLaunchKernel.args__val;
      oss << ", sharedMemBytes=" << data->args.hipExtLaunchKernel.sharedMemBytes;
      oss << ", stream=" << data->args.hipExtLaunchKernel.stream;
      oss << ", startEvent=" << data->args.hipExtLaunchKernel.startEvent;
      oss << ", stopEvent=" << data->args.hipExtLaunchKernel.stopEvent;
      oss << ", flags=" << data->args.hipExtLaunchKernel.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpy2DFromArrayAsync:
      oss << "hipMemcpy2DFromArrayAsync(";
      oss << "dst=" << data->args.hipMemcpy2DFromArrayAsync.dst;
      oss << ", dpitch=" << data->args.hipMemcpy2DFromArrayAsync.dpitch;
      oss << ", src=" << data->args.hipMemcpy2DFromArrayAsync.src;
      oss << ", wOffset=" << data->args.hipMemcpy2DFromArrayAsync.wOffset;
      oss << ", hOffset=" << data->args.hipMemcpy2DFromArrayAsync.hOffset;
      oss << ", width=" << data->args.hipMemcpy2DFromArrayAsync.width;
      oss << ", height=" << data->args.hipMemcpy2DFromArrayAsync.height;
      oss << ", kind=" << data->args.hipMemcpy2DFromArrayAsync.kind;
      oss << ", stream=" << data->args.hipMemcpy2DFromArrayAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetLimit:
      oss << "hipDeviceGetLimit(";
      if (data->args.hipDeviceGetLimit.pValue == NULL) oss << "pValue=NULL";
      else oss << "pValue=" << data->args.hipDeviceGetLimit.pValue__val;
      oss << ", limit=" << data->args.hipDeviceGetLimit.limit;
      oss << ")";
    break;
    case HIP_API_ID_hipModuleLoadDataEx:
      oss << "hipModuleLoadDataEx(";
      if (data->args.hipModuleLoadDataEx.module == NULL) oss << "module=NULL";
      else oss << "module=" << data->args.hipModuleLoadDataEx.module__val;
      oss << ", image=" << data->args.hipModuleLoadDataEx.image;
      oss << ", numOptions=" << data->args.hipModuleLoadDataEx.numOptions;
      if (data->args.hipModuleLoadDataEx.options == NULL) oss << ", options=NULL";
      else oss << ", options=" << data->args.hipModuleLoadDataEx.options__val;
      if (data->args.hipModuleLoadDataEx.optionsValues == NULL) oss << ", optionsValues=NULL";
      else oss << ", optionsValues=" << data->args.hipModuleLoadDataEx.optionsValues__val;
      oss << ")";
    break;
    case HIP_API_ID_hipRuntimeGetVersion:
      oss << "hipRuntimeGetVersion(";
      if (data->args.hipRuntimeGetVersion.runtimeVersion == NULL) oss << "runtimeVersion=NULL";
      else oss << "runtimeVersion=" << data->args.hipRuntimeGetVersion.runtimeVersion__val;
      oss << ")";
    break;
    case HIP_API_ID_hipMemRangeGetAttribute:
      oss << "hipMemRangeGetAttribute(";
      oss << "data=" << data->args.hipMemRangeGetAttribute.data;
      oss << ", data_size=" << data->args.hipMemRangeGetAttribute.data_size;
      oss << ", attribute=" << data->args.hipMemRangeGetAttribute.attribute;
      oss << ", dev_ptr=" << data->args.hipMemRangeGetAttribute.dev_ptr;
      oss << ", count=" << data->args.hipMemRangeGetAttribute.count;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceGetP2PAttribute:
      oss << "hipDeviceGetP2PAttribute(";
      if (data->args.hipDeviceGetP2PAttribute.value == NULL) oss << "value=NULL";
      else oss << "value=" << data->args.hipDeviceGetP2PAttribute.value__val;
      oss << ", attr=" << data->args.hipDeviceGetP2PAttribute.attr;
      oss << ", srcDevice=" << data->args.hipDeviceGetP2PAttribute.srcDevice;
      oss << ", dstDevice=" << data->args.hipDeviceGetP2PAttribute.dstDevice;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyPeerAsync:
      oss << "hipMemcpyPeerAsync(";
      oss << "dst=" << data->args.hipMemcpyPeerAsync.dst;
      oss << ", dstDeviceId=" << data->args.hipMemcpyPeerAsync.dstDeviceId;
      oss << ", src=" << data->args.hipMemcpyPeerAsync.src;
      oss << ", srcDevice=" << data->args.hipMemcpyPeerAsync.srcDevice;
      oss << ", sizeBytes=" << data->args.hipMemcpyPeerAsync.sizeBytes;
      oss << ", stream=" << data->args.hipMemcpyPeerAsync.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipGetDeviceProperties:
      oss << "hipGetDeviceProperties(";
      if (data->args.hipGetDeviceProperties.props == NULL) oss << "props=NULL";
      else oss << "props=" << data->args.hipGetDeviceProperties.props__val;
      oss << ", device=" << data->args.hipGetDeviceProperties.device;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoH:
      oss << "hipMemcpyDtoH(";
      oss << "dst=" << data->args.hipMemcpyDtoH.dst;
      oss << ", src=" << data->args.hipMemcpyDtoH.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyDtoH.sizeBytes;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyWithStream:
      oss << "hipMemcpyWithStream(";
      oss << "dst=" << data->args.hipMemcpyWithStream.dst;
      oss << ", src=" << data->args.hipMemcpyWithStream.src;
      oss << ", sizeBytes=" << data->args.hipMemcpyWithStream.sizeBytes;
      oss << ", kind=" << data->args.hipMemcpyWithStream.kind;
      oss << ", stream=" << data->args.hipMemcpyWithStream.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipDeviceTotalMem:
      oss << "hipDeviceTotalMem(";
      if (data->args.hipDeviceTotalMem.bytes == NULL) oss << "bytes=NULL";
      else oss << "bytes=" << data->args.hipDeviceTotalMem.bytes__val;
      oss << ", device=" << data->args.hipDeviceTotalMem.device;
      oss << ")";
    break;
    case HIP_API_ID_hipHostGetDevicePointer:
      oss << "hipHostGetDevicePointer(";
      if (data->args.hipHostGetDevicePointer.devPtr == NULL) oss << "devPtr=NULL";
      else oss << "devPtr=" << data->args.hipHostGetDevicePointer.devPtr__val;
      oss << ", hstPtr=" << data->args.hipHostGetDevicePointer.hstPtr;
      oss << ", flags=" << data->args.hipHostGetDevicePointer.flags;
      oss << ")";
    break;
    case HIP_API_ID_hipMemRangeGetAttributes:
      oss << "hipMemRangeGetAttributes(";
      if (data->args.hipMemRangeGetAttributes.data == NULL) oss << "data=NULL";
      else oss << "data=" << data->args.hipMemRangeGetAttributes.data__val;
      if (data->args.hipMemRangeGetAttributes.data_sizes == NULL) oss << ", data_sizes=NULL";
      else oss << ", data_sizes=" << data->args.hipMemRangeGetAttributes.data_sizes__val;
      if (data->args.hipMemRangeGetAttributes.attributes == NULL) oss << ", attributes=NULL";
      else oss << ", attributes=" << data->args.hipMemRangeGetAttributes.attributes__val;
      oss << ", num_attributes=" << data->args.hipMemRangeGetAttributes.num_attributes;
      oss << ", dev_ptr=" << data->args.hipMemRangeGetAttributes.dev_ptr;
      oss << ", count=" << data->args.hipMemRangeGetAttributes.count;
      oss << ")";
    break;
    case HIP_API_ID_hipMemcpyParam2D:
      oss << "hipMemcpyParam2D(";
      if (data->args.hipMemcpyParam2D.pCopy == NULL) oss << "pCopy=NULL";
      else oss << "pCopy=" << data->args.hipMemcpyParam2D.pCopy__val;
      oss << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxReset:
      oss << "hipDevicePrimaryCtxReset(";
      oss << "dev=" << data->args.hipDevicePrimaryCtxReset.dev;
      oss << ")";
    break;
    case HIP_API_ID_hipGetMipmappedArrayLevel:
      oss << "hipGetMipmappedArrayLevel(";
      if (data->args.hipGetMipmappedArrayLevel.levelArray == NULL) oss << "levelArray=NULL";
      else oss << "levelArray=" << data->args.hipGetMipmappedArrayLevel.levelArray__val;
      oss << ", mipmappedArray=" << data->args.hipGetMipmappedArrayLevel.mipmappedArray;
      oss << ", level=" << data->args.hipGetMipmappedArrayLevel.level;
      oss << ")";
    break;
    case HIP_API_ID_hipMemsetD32Async:
      oss << "hipMemsetD32Async(";
      oss << "dst=" << data->args.hipMemsetD32Async.dst;
      oss << ", value=" << data->args.hipMemsetD32Async.value;
      oss << ", count=" << data->args.hipMemsetD32Async.count;
      oss << ", stream=" << data->args.hipMemsetD32Async.stream;
      oss << ")";
    break;
    case HIP_API_ID_hipGetDevice:
      oss << "hipGetDevice(";
      if (data->args.hipGetDevice.deviceId == NULL) oss << "deviceId=NULL";
      else oss << "deviceId=" << data->args.hipGetDevice.deviceId__val;
      oss << ")";
    break;
    case HIP_API_ID_hipGetDeviceCount:
      oss << "hipGetDeviceCount(";
      if (data->args.hipGetDeviceCount.count == NULL) oss << "count=NULL";
      else oss << "count=" << data->args.hipGetDeviceCount.count__val;
      oss << ")";
    break;
    case HIP_API_ID_hipIpcOpenEventHandle:
      oss << "hipIpcOpenEventHandle(";
      if (data->args.hipIpcOpenEventHandle.event == NULL) oss << "event=NULL";
      else oss << "event=" << data->args.hipIpcOpenEventHandle.event__val;
      oss << ", handle=" << data->args.hipIpcOpenEventHandle.handle;
      oss << ")";
    break;
    default: oss << "unknown";
  };
  return strdup(oss.str().c_str());
}
#endif  // HIP_PROF_HIP_API_STRING
#endif  // _HIP_PROF_STR_H
