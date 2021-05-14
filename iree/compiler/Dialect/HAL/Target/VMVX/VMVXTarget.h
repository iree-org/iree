// Copyright 2021 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_VMVX_VMVXTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_VMVX_VMVXTARGET_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Options controlling the VM/LA translation.
struct VMVXTargetOptions {
  // TODO(benvanik): target configuration.
  // We'll want things like:
  // - what data types are we supporting (f32, f64, etc)
  // - what version of the ISA are we targeting (v0/v1/v2)
  // - how much scratchpad size can we use (16KB, 32KB, etc)
};

// Returns a VMVXTargetOptions struct initialized with the
// --iree-hal-vm-la-* flags.
VMVXTargetOptions getVMVXTargetOptionsFromFlags();

// Registers the VMVX backends.
void registerVMVXTargetBackends(
    std::function<VMVXTargetOptions()> queryOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_VMVX_VMVXTARGET_H_
