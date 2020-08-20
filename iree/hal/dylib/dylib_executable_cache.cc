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

#include "iree/hal/dylib/dylib_executable_cache.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/dylib/dylib_executable.h"
#include "iree/hal/executable_format.h"

namespace iree {
namespace hal {
namespace dylib {

DyLibExecutableCache::DyLibExecutableCache() = default;

DyLibExecutableCache::~DyLibExecutableCache() = default;

bool DyLibExecutableCache::CanPrepareFormat(ExecutableFormat format) const {
  return format == kExecutableFormatDyLib;
}

StatusOr<ref_ptr<Executable>> DyLibExecutableCache::PrepareExecutable(
    ExecutableLayout* executable_layout, ExecutableCachingModeBitfield mode,
    const ExecutableSpec& spec) {
  IREE_TRACE_SCOPE0("DyLibExecutableCache::PrepareExecutable");

  // TODO(scotttodd): Options for using in-memory files where supported, or not
  //    writing to temp files on disk (and failing if necessary) if not allowed.
  // TODO(scotttodd): Use stable (possibly temp, but reusable) files when
  //    ExecutableCachingMode::AllowPersistentCaching is set. For example,
  //    hash data into a filename and read from / write to GetTempPath() or
  //    GetCachePath() rather than use GetTempFile().

  return DyLibExecutable::Load(spec);
}

}  // namespace dylib
}  // namespace hal
}  // namespace iree
