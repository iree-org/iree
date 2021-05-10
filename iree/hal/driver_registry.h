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

#ifndef IREE_HAL_DRIVER_REGISTRY_H_
#define IREE_HAL_DRIVER_REGISTRY_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/driver.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Factory interface used for driver enumeration and creation.
// The factory is designed to in many cases live in rodata by not requiring any
// real code or processing when the driver is statically known to be available.
// When drivers may be dynamically available based on system configuration a
// factory can discover them and provide them during enumeration.
//
// Delay-loaded drivers that may require non-trivial setup time (such as those
// implemented in dynamic libraries or over RPC) can be speculatively enumerated
// by a factory and then rely on the try_create to actually perform the slow
// work once the user has explicitly signaled that they are willing to pay the
// cost (and deal with the consequences).
//
// WARNING: this API is unstable until the HAL is fully ported. Do not use.
typedef struct {
  // TODO(benvanik): version field.
  IREE_API_UNSTABLE

  // User-defined pointer passed to all functions.
  void* self;

  // Queries the list of available drivers provided by the factory, if any.
  // |out_driver_infos| will be populated with a *reference* to factory data
  // structures (such as the driver name) that callers may choose to clone if
  // needed.
  //
  // Implementers must make their factory enumeration results immutable for the
  // duration they are registered, though the behavior of try_create is allowed
  // to change call-to-call. If a factory needs to mutate its set of enumerated
  // devices then it must do so by first unregistering itself and re-registering
  // only after the changes have been made.
  //
  // Called with the driver registry lock held; may be called from any thread.
  iree_status_t(IREE_API_PTR* enumerate)(
      void* self, const iree_hal_driver_info_t** out_driver_infos,
      iree_host_size_t* out_driver_info_count);

  // Tries to create a driver as previously queried with enumerate.
  // |driver_id| is the opaque ID returned from enumeration; note that there may
  // be a significant amount of time between enumeration and creation and the
  // driver registry lock may have been release between then.
  //
  // Delay-loaded drivers may still fail here if - for example - required system
  // resources are unavailable or permission is denied.
  //
  // Called with the driver registry lock held; may be called from any thread.
  iree_status_t(IREE_API_PTR* try_create)(void* self,
                                          iree_hal_driver_id_t driver_id,
                                          iree_allocator_t allocator,
                                          iree_hal_driver_t** out_driver);
} iree_hal_driver_factory_t;

//===----------------------------------------------------------------------===//
// iree_hal_driver_registry_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_driver_registry_s iree_hal_driver_registry_t;

// Returns the default per-process driver registry.
// In simple applications this is usually where you want to go to register and
// create drivers. More sophisticated applications that want tighter control
// over the visibility of drivers to certain callers such as when dealing with
// requests from multiple users may choose to allocate their own registries and
// manage their lifetime as desired.
//
// TODO(benvanik): remove global registry and make callers manage always. We can
// provide helpers to make that easier to do, but there's really no benefit to
// having this be global like it is. Alternatively, this can be opt-in thanks to
// LTO: if a user doesn't call this then the default registry is never
// allocated.
IREE_API_EXPORT iree_hal_driver_registry_t* iree_hal_driver_registry_default(
    void);

// Registers a driver factory to serve future queries/requests for drivers.
// See iree_hal_driver_registry_t for more information.
//
// Thread-safe. The factory is not retained and must be kept alive by the caller
// until it is unregistered (or the application terminates).
IREE_API_EXPORT iree_status_t iree_hal_driver_registry_register_factory(
    iree_hal_driver_registry_t* registry,
    const iree_hal_driver_factory_t* factory);

// Unregisters a driver factory.
// Unregistering a factory only prevents new drivers from being created;
// existing drivers may remain live even after unregistering. Factories can
// expect that no new drivers will be created via the factory after the call
// returns.
//
// Thread-safe. As the factory is not retained by the registry the caller must
// release its memory (if needed) after this call returns.
IREE_API_EXPORT iree_status_t iree_hal_driver_registry_unregister_factory(
    iree_hal_driver_registry_t* registry,
    const iree_hal_driver_factory_t* factory);

// Enumerates all drivers from registered factories and returns them as a list.
// The provided |allocator| will be used to allocate the returned list and after
// the caller is done with it |out_driver_infos| must be freed with that same
// allocator by the caller.
//
// The set of drivers returned should be considered the superset of those that
// may be available for successful creation as it's possible that delay-loaded
// drivers may fail even if they appear in this list.
//
// Thread-safe. Note that the factory may be unregistered between the query
// completing and any attempt to instantiate the driver.
IREE_API_EXPORT iree_status_t iree_hal_driver_registry_enumerate(
    iree_hal_driver_registry_t* registry, iree_allocator_t allocator,
    iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count);

// Attempts to create a driver registered with the driver registry by a specific
// ID as returned during enumeration in iree_hal_driver_info_t::driver_id.
// This can be used to specify the exact driver to create in cases where there
// may be multiple factories providing drivers with the same name.
//
// Thread-safe. May block the caller if the driver is delay-loaded and needs to
// perform additional loading/verification/etc before returning.
IREE_API_EXPORT iree_status_t iree_hal_driver_registry_try_create(
    iree_hal_driver_registry_t* registry, iree_hal_driver_id_t driver_id,
    iree_allocator_t allocator, iree_hal_driver_t** out_driver);

// Attempts to create a driver registered with the given canonical driver name.
// Effectively enumerate + find by name + try_create if found. Factories are
// searched in most-recently-added order such that it's possible to override
// drivers with newer registrations when multiple factories provide the same
// driver name.
//
// Thread-safe. May block the caller if the driver is delay-loaded and needs to
// perform additional loading/verification/etc before returning.
IREE_API_EXPORT iree_status_t iree_hal_driver_registry_try_create_by_name(
    iree_hal_driver_registry_t* registry, iree_string_view_t driver_name,
    iree_allocator_t allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_REGISTRY_H_
