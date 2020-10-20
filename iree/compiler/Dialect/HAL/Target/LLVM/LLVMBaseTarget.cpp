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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMBaseTarget.h"

#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

LLVMBaseTargetBackend::LLVMBaseTargetBackend(LLVMTargetOptions options)
    : options_(std::move(options)) {}

void LLVMBaseTargetBackend::getDependentDialects(
    DialectRegistry& registry) const {
  // clang-format off
    registry.insert<AffineDialect,
                    linalg::LinalgDialect,
                    LLVM::LLVMDialect,
                    scf::SCFDialect,
                    vector::VectorDialect>();
  // clang-format on
}

void LLVMBaseTargetBackend::buildTranslationPassPipeline(
    ExecutableTargetOp targetOp, OpPassManager& passManager) {
  buildLLVMTransformPassPipeline(passManager);
}

std::array<Value, 3> LLVMBaseTargetBackend::calculateDispatchWorkgroupCount(
    Location loc, IREE::HAL::ExecutableOp executableOp,
    IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
    OpBuilder& builder) {
  // For now we are not tiling and just dispatch everything as 1,1,1.
  auto constantOne = builder.createOrFold<mlir::ConstantIndexOp>(loc, 1);
  return {constantOne, constantOne, constantOne};
}
}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
