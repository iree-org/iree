// Copyright 2019 Google LLC
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

#ifndef IREE_HAL_EXECUTABLE_CACHE_H_
#define IREE_HAL_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_format.h"
#include "iree/hal/executable_layout.h"

namespace iree {
namespace hal {

// A cache of prepared executables for a particular device.
// Caches may be shared across multiple devices from the same driver or specific
// to individual devices. Caches may persist prepared executables across process
// launches or re-prepare them each run. Callers should assume that the cache is
// a no-op and the returned Executables only live for as long as the cache does.
//
// The term 'cache' here is rather optimistic - it's perfectly acceptable for
// implementations to not cache at all and return new Executables for each
// PrepareExecutable called (even for the same executable). Callers should
// expect such behavior and try to retain the results of the PrepareExecutable
// calls to reduce overhead in re-preparing executables.
//
// Thread-safe - multiple threads may prepare executables (including the *same*
// executable) simultaneously.
class ExecutableCache : public RefObject<ExecutableCache> {
 public:
  virtual ~ExecutableCache() = default;

  // TODO(benvanik): status/queries (size, etc).

  // TODO(b/137153339): serialization/deserialization.

  // Returns true if the executable cache can prepare the given executable input
  // format. Preparation may still fail if the particular version or features
  // required by the executable are not supported.
  virtual bool CanPrepareFormat(ExecutableFormat format) const = 0;

  // Prepares an executable for use.
  // The provided |spec| and |executable_data| will be used to either lookup a
  // previously prepared executable in the cache or prepare a new one.
  //
  // Depending on the driver preparation may take a non-trivial amount of time
  // (such as when JITing/etc). As the cache is internally synchronized callers
  // can issue preparation requests from multiple threads - even for the same
  // executables - and calls will block until preparation completes.
  //
  // When preparing a large number of executables it's recommended to use the
  // PrepareExecutables method to batch and wait on the results.
  virtual StatusOr<ref_ptr<Executable>> PrepareExecutable(
      ExecutableLayout* executable_layout,
      iree_hal_executable_caching_mode_t mode,
      iree_const_byte_span_t executable_data) = 0;

 protected:
  ExecutableCache() = default;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_EXECUTABLE_CACHE_H_
