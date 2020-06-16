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
#include "iree/base/dynamic_library.h"
#include "iree/base/status.h"
#include "iree/hal/executable_spec.h"
#include "iree/hal/host/host_executable.h"

namespace iree {
namespace hal {
namespace dylib {

struct MemrefType;

class DyLibExecutable final : public HostExecutable {
 public:
  static StatusOr<ref_ptr<DyLibExecutable>> Load(ExecutableSpec spec);

  DyLibExecutable();
  ~DyLibExecutable() override;

  bool supports_debugging() const override { return false; }

  StatusOr<ref_ptr<DispatchState>> PrepareDispatch(
      const DispatchParams& params) override;
  Status DispatchTile(DispatchState* state,
                      std::array<uint32_t, 3> workgroup_xyz) override;

 private:
  Status Initialize(ExecutableSpec spec);

  std::string executable_library_temp_path_;
  std::unique_ptr<DynamicLibrary> executable_library_;
  absl::InlinedVector<void*, 4> entry_functions_;
};

}  // namespace dylib
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DYLIB_DYLIB_EXECUTABLE_H_
