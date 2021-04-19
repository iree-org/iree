// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_SEMAPHORE_H_
#define IREE_HAL_SEMAPHORE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_s iree_hal_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

// Synchronization mechanism for host->device, device->host, host->host,
// and device->device notification. Semaphores behave like Vulkan timeline
// semaphores (or D3D12 fences) and contain a monotonically increasing
// uint64_t payload. They may be waited on any number of times even if they
// have already been signaled for a particular value. They may also be waited
// on for a particular value prior to the signal for that value.
//
// A semaphore is updated to its new value after all prior commands have
// completed but the delay between completion and the host being woken varies.
// Some implementations may coalesce semaphores to avoid spurious waking while
// others will immediately synchronize with the host.
//
// One use of semaphores is for resource lifetime management: all resources used
// by a set of submission batches must be considered live until the semaphore
// attached to the submission has signaled.
//
// Another use of semaphores is device->device synchronization for setting up
// the DAG of command buffers across queue submissions. This allows devices to
// perform non-trivial scheduling behavior without the need to wake the host.
//
// Semaphores may be set to a permanently failed state by implementations when
// errors occur during asynchronous execution. Users are expected to propagate
// the failures and possibly reset the entire device that produced the error.
//
// For more information on semaphores see the following docs describing how
// timelines are generally used (specifically in the device->host case):
// https://www.youtube.com/watch?v=SpE--Rf516Y
// https://www.khronos.org/assets/uploads/developers/library/2018-xdc/Vulkan-Timeline-Semaphores-Part-1_Sep18.pdf
// https://docs.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization
typedef struct iree_hal_semaphore_s iree_hal_semaphore_t;

// Creates a semaphore that can be used with command queues owned by this
// device. To use the semaphores with other devices or instances they must
// first be exported.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_create(iree_hal_device_t* device, uint64_t initial_value,
                          iree_hal_semaphore_t** out_semaphore);

// Retains the given |semaphore| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_retain(iree_hal_semaphore_t* semaphore);

// Releases the given |semaphore| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_release(iree_hal_semaphore_t* semaphore);

// Queries the current payload of the semaphore and stores the result in
// |out_value|. As the payload is monotonically increasing it is guaranteed that
// the value is at least equal to the previous result of a
// iree_hal_semaphore_query call and coherent with any waits for a
// specified value via iree_device_wait_all_semaphores.
//
// Returns the status at the time the method is called without blocking and as
// such is only valid after a semaphore has been signaled. The same failure
// status will be returned regardless of when in the timeline the error
// occurred.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_query(iree_hal_semaphore_t* semaphore, uint64_t* out_value);

// Signals the |semaphore| to the given payload value.
// The call is ignored if the current payload value exceeds |new_value|.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_signal(iree_hal_semaphore_t* semaphore, uint64_t new_value);

// Signals the |semaphore| with a failure. The |status| will be returned from
// iree_hal_semaphore_query and iree_hal_semaphore_signal for the lifetime
// of the semaphore. Ownership of the status transfers to the semaphore and
// callers must clone it if they wish to retain it.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_fail(iree_hal_semaphore_t* semaphore, iree_status_t status);

// Blocks the caller until the semaphore reaches or exceedes the specified
// payload value or the |timeout| elapses.
//
// Returns success if the wait is successful and the semaphore has met or
// exceeded the required payload value.
//
// Returns DEADLINE_EXCEEDED if the |timeout| elapses without the semaphore
// reaching the required value. If an asynchronous failure occured this will
// return the failure status that was set immediately.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_semaphore_wait(
    iree_hal_semaphore_t* semaphore, uint64_t value, iree_timeout_t timeout);

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t implementation details
//===----------------------------------------------------------------------===//

typedef struct {
  // << HAL C porting in progress >>
  IREE_API_UNSTABLE

  void(IREE_API_PTR* destroy)(iree_hal_semaphore_t* semaphore);

  iree_status_t(IREE_API_PTR* query)(iree_hal_semaphore_t* semaphore,
                                     uint64_t* out_value);
  iree_status_t(IREE_API_PTR* signal)(iree_hal_semaphore_t* semaphore,
                                      uint64_t new_value);
  void(IREE_API_PTR* fail)(iree_hal_semaphore_t* semaphore,
                           iree_status_t status);

  iree_status_t(IREE_API_PTR* wait)(iree_hal_semaphore_t* semaphore,
                                    uint64_t value, iree_timeout_t timeout);
} iree_hal_semaphore_vtable_t;

IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_destroy(iree_hal_semaphore_t* semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_SEMAPHORE_H_
