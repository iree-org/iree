// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-codegen-rocdl-configure-buffer-instructions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLCONFIGUREBUFFERINSTRUCTIONSPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

static llvm::cl::opt<bool> clROCDLlEnableBufferInstructions(
    "iree-rocdl-enable-buffer-instructions",
    llvm::cl::desc("Use buffer instructions (by using buffer fat pointers) "
                   "where possible on AMD targets"),
    llvm::cl::init(true));

static Value stripIntegerCasts(Value val) {
  while (isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp,
                         arith::ExtSIOp, arith::ExtUIOp>(val.getDefiningOp())) {
    val = val.getDefiningOp()->getOperand(0);
  }
  return val;
}

/// Determine if `arg` is an arithmetic function of constants or HAL constant
/// loads, which is a conservative approximatino for workgroup-uniformity that
/// can be made more extensive if needed.
static bool isDefinitelyWorkgroupUniform(Value arg) {
  if (!arg) {
    return true;
  }
  SetVector<Operation *> dependencies;
  BackwardSliceOptions opts;
  arg = stripIntegerCasts(arg);
  if (auto assume = arg.getDefiningOp<IREE::Util::AssumeIntOp>()) {
    arg = assume.getOperand(cast<OpResult>(arg).getResultNumber());
  }
  // Note: this is a bit conservative, in that it will traverse all the
  // arguments to a util.assume.int that isn't the immediate parent of val.
  [[maybe_unused]] LogicalResult result =
      getBackwardSlice(arg, &dependencies, opts);
  assert(result.succeeded());
  return llvm::all_of(dependencies, [&](Operation *op) {
    if (matchPattern(op, m_Constant())) {
      return true;
    }
    if (isa<IREE::HAL::InterfaceConstantLoadOp, IREE::Util::AssumeIntOp>(op)) {
      return true;
    }
    if (isa_and_nonnull<arith::ArithDialect>(op->getDialect())) {
      return true;
    }
    return false;
  });
}

namespace {

struct ROCDLConfigureBufferInstructionsPass final
    : impl::ROCDLConfigureBufferInstructionsPassBase<
          ROCDLConfigureBufferInstructionsPass> {
  void runOnOperation() override {
    if (!clROCDLlEnableBufferInstructions) {
      return;
    }
    mlir::FunctionOpInterface funcOp = getOperation();
    // Is this really the best way to skip this pass on non-rocdl targets?
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target || !target.isAMD()) {
      return;
    }

    // Initialize the DataFlowSolver with IntegerRangeAnalysis.
    DataFlowSolver solver;
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(funcOp))) {
      funcOp.emitOpError("failed to run integer range analysis");
      return signalPassFailure();
    }

    auto *gpuDialect =
        getContext().getLoadedDialect<IREE::GPU::IREEGPUDialect>();
    auto annotationHelper =
        gpuDialect->getUseRocdlBufferInstructionsAttrHelper();
    auto unitAttr = UnitAttr::get(&getContext());
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp binding) {
      Value offset = binding.getByteOffset();
      if (offset && !isDefinitelyWorkgroupUniform(offset)) {
        LDBG() << "Binding offset " << offset
               << " not known workgroup-uniform\n";
        return;
      }
      ShapedType resultTy = dyn_cast<ShapedType>(binding.getType());
      if (auto tensorType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
              binding.getType())) {
        resultTy = tensorType.asRankedTensorType();
      }
      if (!canUseFatRawBuffer(resultTy, binding.getResultDynamicDims(0),
                              &solver)) {
        LDBG() << "Size of " << binding << " too large or unbounded";
        return;
      }
      annotationHelper.setAttr(binding, unitAttr);
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler
