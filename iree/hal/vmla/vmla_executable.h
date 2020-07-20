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

#ifndef IREE_HAL_VMLA_VMLA_EXECUTABLE_H_
#define IREE_HAL_VMLA_VMLA_EXECUTABLE_H_

#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/hal/executable_spec.h"
#include "iree/hal/host/host_executable.h"
#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"

namespace iree {
namespace hal {
namespace vmla {

class Interface;

class VMLAExecutable final : public HostExecutable {
 public:
  static StatusOr<ref_ptr<VMLAExecutable>> Load(iree_vm_instance_t* instance,
                                                iree_vm_module_t* vmla_module,
                                                ExecutableSpec spec,
                                                bool allow_aliasing_data);

  VMLAExecutable(ExecutableSpec spec, bool allow_aliasing_data);
  ~VMLAExecutable() override;

  bool supports_debugging() const override { return false; }

  // Reference to the bytecode blob contents.
  absl::Span<const uint8_t> executable_data() const {
    return spec_.executable_data;
  }

  // VM context containing the loaded executable module.
  iree_vm_context_t* context() const { return context_; }

  // Entry point functions in export order.
  absl::Span<const iree_vm_function_t> entry_functions() const {
    return absl::MakeConstSpan(entry_functions_);
  }

  StatusOr<ref_ptr<DispatchState>> PrepareDispatch(
      const DispatchParams& params) override;
  Status DispatchTile(DispatchState* state,
                      std::array<uint32_t, 3> workgroup_xyz) override;

 private:
  Status Initialize(iree_vm_instance_t* instance,
                    iree_vm_module_t* vmla_module);

  ExecutableSpec spec_;
  std::vector<uint8_t> cloned_executable_data_;

  iree_vm_context_t* context_ = nullptr;
  absl::InlinedVector<iree_vm_function_t, 4> entry_functions_;
};

}  // namespace vmla
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VMLA_VMLA_EXECUTABLE_H_
