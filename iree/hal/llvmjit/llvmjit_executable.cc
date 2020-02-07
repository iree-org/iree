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

#include "iree/hal/llvmjit/llvmjit_executable.h"

#include <iostream>

namespace iree {
namespace hal {
namespace llvmjit {

// static
StatusOr<ref_ptr<LLVMJITExecutable>> LLVMJITExecutable::Load(
    hal::Allocator* allocator, ExecutableSpec spec, bool allow_aliasing_data) {
  // Allocate the executable now.
  // We do this here so that if we need to clone the data we are passing that
  // to the VM loader instead of the data we may not have access to later.

  auto executable =
      make_ref<LLVMJITExecutable>(allocator, spec, allow_aliasing_data);

  return executable;
}

LLVMJITExecutable::LLVMJITExecutable(hal::Allocator* allocator,
                                     ExecutableSpec spec,
                                     bool allow_aliasing_data)
    : spec_(spec) {
  if (!allow_aliasing_data) {
    // Clone data.
    cloned_executable_data_ = {spec.executable_data.begin(),
                               spec.executable_data.end()};
    spec_.executable_data = absl::MakeConstSpan(cloned_executable_data_);
  }
}

LLVMJITExecutable::~LLVMJITExecutable() = default;
}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
