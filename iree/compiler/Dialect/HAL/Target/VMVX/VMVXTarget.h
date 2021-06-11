// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
