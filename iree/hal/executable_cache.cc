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

#include "iree/hal/executable_cache.h"

#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {

ExecutableCache::ExecutableCache() = default;

ExecutableCache::~ExecutableCache() = default;

StatusOr<WaitHandle> ExecutableCache::PrepareExecutables(
    ExecutableCachingModeBitfield mode, absl::Span<const ExecutableSpec> specs,
    absl::Span<ref_ptr<Executable>> out_executables) {
  IREE_TRACE_SCOPE0("ExecutableCache::PrepareExecutables");
  if (specs.size() != out_executables.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "1:1 specs:out_executables required";
  }

  ManualResetEvent fence("ExecutableCachePreparation");
  auto wait_handle = fence.OnSet();

  // TODO(benvanik): make async (spin up thread, etc).
  for (int i = 0; i < specs.size(); ++i) {
    auto executable_or = PrepareExecutable(mode, specs[i]);
    if (!executable_or.ok()) {
      // TODO(benvanik): propagate executable error.
      RETURN_IF_ERROR(fence.Set());
      return wait_handle;
    }
    out_executables[i] = add_ref(std::move(executable_or).ValueOrDie());
  }

  RETURN_IF_ERROR(fence.Set());
  return wait_handle;
}

}  // namespace hal
}  // namespace iree
