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

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_ROCM_ROCMTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_ROCM_ROCMTARGET_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "llvm/IR/Module.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

struct ROCMTargetOptions {
  // ROCm target Chip
  std::string ROCMTargetChip;
  // Whether to try Linking to AMD Bitcodes
  bool ROCMLinkBC;
};

ROCMTargetOptions getROCMTargetOptionsFromFlags();

// Registers the ROCM backend.
void registerROCMTargetBackends(
    std::function<ROCMTargetOptions()> queryOptions);

// Links LLVM module to ROC Device Library Bit Code
void LinkROCDLIfNecessary(llvm::Module *module);

// Compiles ISAToHsaco Code
std::string createHsaco(const std::string isa, StringRef name);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_ROCM_ROCMTARGET_H_
