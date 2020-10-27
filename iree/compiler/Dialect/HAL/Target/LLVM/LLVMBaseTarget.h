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

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMBASETARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMBASETARGET_H_

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Base target for LLVM ahead-of-time (AOT) and just-in-time (JIT) backends.
class LLVMBaseTargetBackend : public TargetBackend {
 public:
  explicit LLVMBaseTargetBackend(LLVMTargetOptions options);

  void getDependentDialects(DialectRegistry &registry) const override;

  void buildTranslationPassPipeline(ExecutableTargetOp targetOp,
                                    OpPassManager &passManager) override;

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override;

  LogicalResult recordDispatch(Location loc, DispatchState dispatchState,
                               DeviceSwitchRewriter &switchRewriter) override;

 protected:
  LLVMTargetOptions options_;
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMBASETARGET_H_
