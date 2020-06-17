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

#include "iree/hal/llvmjit/llvmjit_executable_cache.h"

#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/executable_format.h"
#include "iree/hal/llvmjit/llvmjit_executable.h"

namespace iree {
namespace hal {
namespace llvmjit {

LLVMJITExecutableCache::LLVMJITExecutableCache() = default;

LLVMJITExecutableCache::~LLVMJITExecutableCache() = default;

bool LLVMJITExecutableCache::CanPrepareFormat(ExecutableFormat format) const {
  return format == kExecutableFormatLLVM;
}

StatusOr<ref_ptr<Executable>> LLVMJITExecutableCache::PrepareExecutable(
    ExecutableLayout* executable_layout, ExecutableCachingModeBitfield mode,
    const ExecutableSpec& spec) {
  IREE_TRACE_SCOPE0("LLVMJITExecutableCache::PrepareExecutable");

  // Wrap the data (or copy it).
  bool allow_aliasing_data =
      AllBitsSet(mode, ExecutableCachingMode::kAliasProvidedData);
  ASSIGN_OR_RETURN(auto executable,
                   LLVMJITExecutable::Load(spec, !allow_aliasing_data));

  return executable;
}

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
