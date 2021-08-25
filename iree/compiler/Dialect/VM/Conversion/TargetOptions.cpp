// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

TargetOptions getTargetOptionsFromFlags() {
  static llvm::cl::OptionCategory vmTargetOptionsCategory(
      "IREE VM target options");

  static auto *indexBitsFlag = new llvm::cl::opt<int>{
      "iree-vm-target-index-bits",
      llvm::cl::init(32),
      llvm::cl::desc("Bit width of index types."),
      llvm::cl::cat(vmTargetOptionsCategory),
  };
  static auto *i64ExtensionFlag = new llvm::cl::opt<bool>{
      "iree-vm-target-extension-i64",
      llvm::cl::init(false),
      llvm::cl::desc("Support i64 target opcode extensions."),
      llvm::cl::cat(vmTargetOptionsCategory),
  };
  static auto *f32ExtensionFlag = new llvm::cl::opt<bool>{
      "iree-vm-target-extension-f32",
      llvm::cl::init(true),
      llvm::cl::desc("Support f32 target opcode extensions."),
      llvm::cl::cat(vmTargetOptionsCategory),
  };
  static auto *f64ExtensionFlag = new llvm::cl::opt<bool>{
      "iree-vm-target-extension-f64",
      llvm::cl::init(false),
      llvm::cl::desc("Support f64 target opcode extensions."),
      llvm::cl::cat(vmTargetOptionsCategory),
  };
  static auto *truncateUnsupportedIntegersFlag = new llvm::cl::opt<bool>{
      "iree-vm-target-truncate-unsupported-integers",
      llvm::cl::init(true),
      llvm::cl::desc("Truncate i64 to i32 when unsupported."),
      llvm::cl::cat(vmTargetOptionsCategory),
  };
  static auto *truncateUnsupportedFloatsFlag = new llvm::cl::opt<bool>{
      "iree-vm-target-truncate-unsupported-floats",
      llvm::cl::init(true),
      llvm::cl::desc("Truncate f64 to f32 when unsupported."),
      llvm::cl::cat(vmTargetOptionsCategory),
  };

  TargetOptions targetOptions;
  targetOptions.indexBits = *indexBitsFlag;
  if (*i64ExtensionFlag) {
    targetOptions.i64Extension = true;
  }
  if (*f32ExtensionFlag) {
    targetOptions.f32Extension = true;
  }
  if (*f64ExtensionFlag) {
    targetOptions.f64Extension = true;
  }
  targetOptions.truncateUnsupportedIntegers = *truncateUnsupportedIntegersFlag;
  targetOptions.truncateUnsupportedFloats = *truncateUnsupportedFloatsFlag;
  return targetOptions;
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
