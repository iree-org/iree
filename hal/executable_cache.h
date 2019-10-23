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

#include "base/bitfield.h"
#include "base/ref_ptr.h"
#include "base/status.h"
#include "hal/executable.h"
#include "hal/executable_format.h"
#include "hal/executable_spec.h"

namespace iree {
namespace hal {

// Defines how the executable cache performs preparation.
enum class ExecutableCachingMode : uint32_t {
  // Allows the cache to reference the provided executable_data after it has
  // prepared the executable. Callers must ensure the data remains valid for the
  // lifetime of the cache. If memory mapping constant executable data from
  // disk this can be used to avoid copies.
  kAliasProvidedData = 1 << 0,

  // Allows the prepared executable to be cached persistently (on disk/etc).
  // Enable for any executable that is likely to be used in future runs.
  // Note that not all caches support persistent serialization and this is just
  // a hint.
  kAllowPersistentCaching = 1 << 1,

  // Allows the cache to optimize the executable as much as it can.
  // This may cause preparation to take significantly longer while (hopefully)
  // improving runtime performance. Avoid for one-shot executables.
  kAllowOptimization = 1 << 2,

  // Enables Executable debugging methods if supported by the device and
  // executable. This may disable certain optimizations or retain additional
  // data to allow disassembly, stepping, etc.
  //
  // Device must support the DeviceFeature::kDebugging feature and executables
  // must support the ExecutableFeature::kDebugging feature.
  kEnableDebugging = 1 << 3,

  // Enables Executable coverage if supported by the device and executable.
  // Depending on the optimization mode this may produce partial coverage
  // results (for example, when certain source operations were optimized away).
  //
  // Device must support the DeviceFeature::kCoverage feature and executables
  // must support the ExecutableFeature::kCoverage feature.
  kEnableCoverage = 1 << 4,

  // Enables Executable profiling if supported by the device and executable.
  // Depending on the optimization mode this may produce partial profiling
  // results. Profiling attribution (whether to the entire executable or
  // specific operations) depends on the implementation.
  //
  // Device must support the DeviceFeature::kProfiling feature and executables
  // must support the ExecutableFeature::kProfiling feature.
  kEnableProfiling = 1 << 5,

  // Default caching mode.
  kDefault = kAllowPersistentCaching | kAllowOptimization,
};
IREE_BITFIELD(ExecutableCachingMode);
using ExecutableCachingModeBitfield = ExecutableCachingMode;

// A cache of prepared executables for a particular device.
// Caches may be shared across multiple devices from the same driver or specific
// to individual devices. Caches may persist prepared executables across process
// launches or reprepare them each run. Callers should assume that the cache is
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
class ExecutableCache {
 public:
  virtual ~ExecutableCache();

  // TODO(benvanik): status/queries (size, etc).

  // TODO(b/137153339): serialization/deserialization.

  // Returns true if the executable cache can prepare the given executable input
  // format. Perparation may still fail if the particular version or features
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
      ExecutableCachingModeBitfield mode, const ExecutableSpec& spec) = 0;

 protected:
  ExecutableCache();
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_EXECUTABLE_CACHE_H_
