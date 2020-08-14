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
      llvm::cl::desc("Bit width of index types"),
      llvm::cl::cat(vmTargetOptionsCategory),
  };
  static auto *extensionsFlag = new llvm::cl::list<OpcodeExtension>{
      "iree-vm-target-extensions",
      llvm::cl::ZeroOrMore,
      llvm::cl::desc("Supported target opcode extensions"),
      llvm::cl::cat(vmTargetOptionsCategory),
      llvm::cl::values(
          clEnumValN(OpcodeExtension::kI64, "i64", "i64 type support")),
  };
  static auto *truncateUnsupportedIntegersFlag = new llvm::cl::opt<bool>{
      "iree-vm-target-truncate-unsupported-integers",
      llvm::cl::init(true),
      llvm::cl::desc("Truncate i64 to i32 when unsupported"),
      llvm::cl::cat(vmTargetOptionsCategory),
  };

  TargetOptions targetOptions;
  targetOptions.indexBits = *indexBitsFlag;
  for (auto ext : *extensionsFlag) {
    switch (ext) {
      case OpcodeExtension::kI64:
        targetOptions.i64Extension = true;
        break;
    }
  }
  targetOptions.truncateUnsupportedIntegers = *truncateUnsupportedIntegersFlag;
  return targetOptions;
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
