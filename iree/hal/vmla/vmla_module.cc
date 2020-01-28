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

#include "iree/hal/vmla/vmla_module.h"

#include "absl/types/span.h"
#include "iree/base/tracing.h"
#include "iree/vm/module_abi_cc.h"
#include "iree/vm/module_abi_packing.h"

namespace iree {
namespace hal {
namespace vmla {
namespace {

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vmla_module_register_types() {
  static bool has_registered = false;
  if (has_registered) return IREE_STATUS_OK;

  has_registered = true;
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// Type wrappers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Module state and method implementation
//===----------------------------------------------------------------------===//

// Per-executable VMLA module state.
// This provides the exported kernel functions to the VM and is instantiated
// one or more times per executable used within a device. Any state here can be
// treated as workgroup-local memory.
//
// Thread-compatible.
class VMLAModuleState final {
 public:
  VMLAModuleState(iree_allocator_t allocator) : allocator_(allocator) {}

  ~VMLAModuleState() = default;

  Status Dummy() { return OkStatus(); }

 private:
  iree_allocator_t allocator_;
};

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

static const vm::NativeFunction<VMLAModuleState> kVMLAModuleFunctions[] = {
    vm::MakeNativeFunction("vmla.dummy", &VMLAModuleState::Dummy),
};

// Per-device VMLA module.
// One of these will be created per device and be shared across all executables
// that are created within that device. Large shared kernel state can go here
// (such as thread pools/caches/etc), though note that they must be either
// thread-safe or internally synchronized.
//
// Thread-safe.
class VMLAModule final : public vm::NativeModule<VMLAModuleState> {
 public:
  explicit VMLAModule(iree_allocator_t allocator)
      : vm::NativeModule<VMLAModuleState>(
            "vmla", allocator, absl::MakeConstSpan(kVMLAModuleFunctions)) {}
  ~VMLAModule() = default;

  Status Initialize() {
    IREE_TRACE_SCOPE0("VMLAModule::Initialize");

    return OkStatus();
  }

  StatusOr<std::unique_ptr<VMLAModuleState>> CreateState(
      iree_allocator_t allocator) override {
    IREE_TRACE_SCOPE0("VMLAModule::CreateState");
    auto state = std::make_unique<VMLAModuleState>(allocator);
    return state;
  }

 private:
};

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vmla_module_create(
    iree_allocator_t allocator, iree_vm_module_t** out_module) {
  if (!out_module) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module = nullptr;
  auto module = std::make_unique<VMLAModule>(allocator);
  IREE_API_RETURN_IF_ERROR(module->Initialize());
  *out_module = module.release()->interface();
  return IREE_STATUS_OK;
}

}  // namespace
}  // namespace vmla
}  // namespace hal
}  // namespace iree
