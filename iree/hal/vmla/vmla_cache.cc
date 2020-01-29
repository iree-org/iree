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

#include "iree/hal/vmla/vmla_cache.h"

#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/executable_format.h"
#include "iree/hal/vmla/vmla_executable.h"

namespace iree {
namespace hal {
namespace vmla {

VMLACache::VMLACache(hal::Allocator* allocator) : allocator_(allocator) {}

VMLACache::~VMLACache() = default;

bool VMLACache::CanPrepareFormat(ExecutableFormat format) const {
  return format == kExecutableFormatVMLA;
}

StatusOr<ref_ptr<Executable>> VMLACache::PrepareExecutable(
    ExecutableCachingModeBitfield mode, const ExecutableSpec& spec) {
  IREE_TRACE_SCOPE0("VMLACache::PrepareExecutable");
  if (!CanPrepareFormat(spec.format)) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Unsupported format: " << spec.format;
  }

  // Wrap the data (or copy it).
  bool allow_aliasing_data =
      AllBitsSet(mode, ExecutableCachingMode::kAliasProvidedData);
  ASSIGN_OR_RETURN(auto executable, VMLAExecutable::Load(allocator_, spec,
                                                         !allow_aliasing_data));

  return executable;
}

}  // namespace vmla
}  // namespace hal
}  // namespace iree
