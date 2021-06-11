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
  static auto *extensionsFlag = new llvm::cl::list<OpcodeExtension>{
      "iree-vm-target-extension",
      llvm::cl::ZeroOrMore,
      llvm::cl::desc("Supported target opcode extensions."),
      llvm::cl::cat(vmTargetOptionsCategory),
      llvm::cl::values(
          clEnumValN(OpcodeExtension::kI64, "i64", "i64 type support"),
          clEnumValN(OpcodeExtension::kF32, "f32", "f32 type support"),
          clEnumValN(OpcodeExtension::kF64, "f64", "f64 type support")),
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
  for (auto ext : *extensionsFlag) {
    switch (ext) {
      case OpcodeExtension::kI64:
        targetOptions.i64Extension = true;
        break;
      case OpcodeExtension::kF32:
        targetOptions.f32Extension = true;
        break;
      case OpcodeExtension::kF64:
        targetOptions.f64Extension = true;
        break;
    }
  }
  targetOptions.truncateUnsupportedIntegers = *truncateUnsupportedIntegersFlag;
  targetOptions.truncateUnsupportedFloats = *truncateUnsupportedFloatsFlag;
  return targetOptions;
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
