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

#ifndef IREE_HAL_DYLIB_DYLIB_EXECUTABLE_H_
#define IREE_HAL_DYLIB_DYLIB_EXECUTABLE_H_

#include <memory>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/base/dynamic_library.h"
#include "iree/base/status.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_spec.h"

namespace iree {
namespace hal {
namespace dylib {

struct MemrefType;

class DyLibExecutable final : public Executable {
 public:
  static StatusOr<ref_ptr<DyLibExecutable>> Load(ExecutableSpec spec);

  DyLibExecutable(ExecutableSpec spec);
  ~DyLibExecutable() override;

  bool supports_debugging() const override { return false; }

  Status Invoke(int func_id, absl::Span<void*> args) const;

 private:
  Status Initialize();

  ExecutableSpec spec_;
  std::string executable_library_temp_path_;
  std::unique_ptr<DynamicLibrary> executable_library_;
  absl::InlinedVector<void*, 4> entry_functions_;
};

}  // namespace dylib
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DYLIB_DYLIB_EXECUTABLE_H_
