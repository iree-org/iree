// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/status_util.h"

#include <stddef.h>

#include "iree/base/status.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"

// The list of HIP error strings with their corresponding IREE error state
// classification.
//
// Note that the list of errors is taken from `hipError_t` enum in
// hip_runtime_api.h. This may not be an exhaustive list;
// common ones here.
#define IREE_HIP_ERROR_LIST(IREE_HIP_MAP_ERROR)                                \
  IREE_HIP_MAP_ERROR("hipSuccess", IREE_STATUS_OK)                             \
  IREE_HIP_MAP_ERROR("hipErrorInvalidValue", IREE_STATUS_INVALID_ARGUMENT)     \
  IREE_HIP_MAP_ERROR("hipErrorOutOfMemory", IREE_STATUS_RESOURCE_EXHAUSTED)    \
  IREE_HIP_MAP_ERROR("hipErrorMemoryAllocation",                               \
                     IREE_STATUS_RESOURCE_EXHAUSTED)                           \
  IREE_HIP_MAP_ERROR("hipErrorNotInitialized", IREE_STATUS_INTERNAL)           \
  IREE_HIP_MAP_ERROR("hipErrorInitializationError", IREE_STATUS_INTERNAL)      \
  IREE_HIP_MAP_ERROR("hipErrorDeinitialized", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorInvalidConfiguration",                           \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidPitchValue",                              \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidSymbol", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorInvalidDevicePointer",                           \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidMemcpyDirection",                         \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInsufficientDriver",                             \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorMissingConfiguration",                           \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorPriorLaunchFailure",                             \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidDeviceFunction", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorNoDevice", IREE_STATUS_FAILED_PRECONDITION)      \
  IREE_HIP_MAP_ERROR("hipErrorInvalidDevice", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorInvalidImage", IREE_STATUS_FAILED_PRECONDITION)  \
  IREE_HIP_MAP_ERROR("hipErrorInvalidContext",                                 \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorContextAlreadyCurrent",                          \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorMapFailed", IREE_STATUS_INTERNAL)                \
  IREE_HIP_MAP_ERROR("hipErrorMapBufferObjectFailed", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorUnmapFailed", IREE_STATUS_INTERNAL)              \
  IREE_HIP_MAP_ERROR("hipErrorArrayIsMapped", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorAlreadyMapped", IREE_STATUS_ALREADY_EXISTS)      \
  IREE_HIP_MAP_ERROR("hipErrorNoBinaryForGpu", IREE_STATUS_INTERNAL)           \
  IREE_HIP_MAP_ERROR("hipErrorAlreadyAcquired", IREE_STATUS_ALREADY_EXISTS)    \
  IREE_HIP_MAP_ERROR("hipErrorNotMapped", IREE_STATUS_INTERNAL)                \
  IREE_HIP_MAP_ERROR("hipErrorNotMappedAsArray", IREE_STATUS_INTERNAL)         \
  IREE_HIP_MAP_ERROR("hipErrorNotMappedAsPointer", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorECCNotCorrectable", IREE_STATUS_DATA_LOSS)       \
  IREE_HIP_MAP_ERROR("hipErrorUnsupportedLimit", IREE_STATUS_INTERNAL)         \
  IREE_HIP_MAP_ERROR("hipErrorContextAlreadyInUse",                            \
                     IREE_STATUS_ALREADY_EXISTS)                               \
  IREE_HIP_MAP_ERROR("hipErrorPeerAccessUnsupported", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorInvalidKernelFile", IREE_STATUS_INTERNAL)        \
  IREE_HIP_MAP_ERROR("hipErrorInvalidGraphicsContext", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorInvalidGraphicsContext", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorInvalidSource", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorSharedObjectSymbolNotFound",                     \
                     IREE_STATUS_NOT_FOUND)                                    \
  IREE_HIP_MAP_ERROR("hipErrorSharedObjectInitFailed",                         \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorOperatingSystem", IREE_STATUS_INTERNAL)          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidHandle", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorInvalidResourceHandle",                          \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorIllegalState", IREE_STATUS_INTERNAL)             \
  IREE_HIP_MAP_ERROR("hipErrorNotFound", IREE_STATUS_NOT_FOUND)                \
  IREE_HIP_MAP_ERROR("hipErrorNotReady", IREE_STATUS_UNAVAILABLE)              \
  IREE_HIP_MAP_ERROR("hipErrorIllegalAddress", IREE_STATUS_INTERNAL)           \
  IREE_HIP_MAP_ERROR("hipErrorLaunchOutOfResources",                           \
                     IREE_STATUS_RESOURCE_EXHAUSTED)                           \
  IREE_HIP_MAP_ERROR("hipErrorLaunchTimeOut", IREE_STATUS_DEADLINE_EXCEEDED)   \
  IREE_HIP_MAP_ERROR("hipErrorPeerAccessAlreadyEnabled",                       \
                     IREE_STATUS_ALREADY_EXISTS)                               \
  IREE_HIP_MAP_ERROR("hipErrorPeerAccessNotEnabled", IREE_STATUS_INTERNAL)     \
  IREE_HIP_MAP_ERROR("hipErrorSetOnActiveProcess", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorContextIsDestroyed", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorAssert", IREE_STATUS_INTERNAL)                   \
  IREE_HIP_MAP_ERROR("hipErrorHostMemoryAlreadyRegistered",                    \
                     IREE_STATUS_ALREADY_EXISTS)                               \
  IREE_HIP_MAP_ERROR("hipErrorHostMemoryNotRegistered", IREE_STATUS_INTERNAL)  \
  IREE_HIP_MAP_ERROR("hipErrorLaunchFailure", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorCooperativeLaunchTooLarge",                      \
                     IREE_STATUS_INTERNAL)                                     \
  IREE_HIP_MAP_ERROR("hipErrorNotSupported", IREE_STATUS_UNAVAILABLE)          \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureUnsupported", IREE_STATUS_INTERNAL) \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureInvalidated", IREE_STATUS_INTERNAL) \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureMerge", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureUnmatched", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureUnjoined", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureIsolation", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureImplicit", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorCapturedEvent", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureWrongThread", IREE_STATUS_INTERNAL) \
  IREE_HIP_MAP_ERROR("hipErrorGraphExecUpdateFailure", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorUnknown", IREE_STATUS_UNKNOWN)                   \
  IREE_HIP_MAP_ERROR("hipErrorRuntimeMemory", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorRuntimeOther", IREE_STATUS_INTERNAL)

// Converts HIP |error_name| to the corresponding IREE status code.
static iree_status_code_t iree_hal_hip_error_name_to_status_code(
    const char* error_name) {
#define IREE_HIP_ERROR_TO_IREE_STATUS(hip_error, iree_status)   \
  if (strncmp(error_name, hip_error, strlen(hip_error)) == 0) { \
    return iree_status;                                         \
  }
  IREE_HIP_ERROR_LIST(IREE_HIP_ERROR_TO_IREE_STATUS)
#undef IREE_HIP_ERROR_TO_IREE_STATUS
  return IREE_STATUS_UNKNOWN;
}

iree_status_t iree_hal_hip_result_to_status(
    const iree_hal_hip_dynamic_symbols_t* syms, hipError_t result,
    const char* file, uint32_t line) {
  if (IREE_LIKELY(result == hipSuccess)) {
    return iree_ok_status();
  }

  const char* error_name = syms->hipGetErrorName(result);
  if (result == hipErrorUnknown) {
    error_name = "HIP_ERROR_UNKNOWN";
  }

  const char* error_string = syms->hipGetErrorString(result);
  if (result == hipErrorUnknown) {
    error_string = "unknown error";
  }

  return iree_make_status_with_location(
      file, line, iree_hal_hip_error_name_to_status_code(error_name),
      "HIP driver error '%s' (%d): %s", error_name, result, error_string);
}
