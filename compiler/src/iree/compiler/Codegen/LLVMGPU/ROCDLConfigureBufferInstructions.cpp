// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Analysis/ROCDLThreadUniformAnalysis.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-codegen-rocdl-configure-buffer-instructions"
#define LDBGS(X) LLVM_DEBUG((llvm::dbgs() << "[" DEBUG_TYPE "]:") << X << "\n")

namespace mlir::iree_compiler {
static llvm::cl::opt<bool> clROCDLlEnableBufferInstructions(
    "iree-rocdl-enable-buffer-instructions",
    llvm::cl::desc("Use buffer instructions (by using buffer fat pointers) "
                   "where possible on AMD targets"),
    llvm::cl::init(true));

#define GEN_PASS_DEF_ROCDLCONFIGUREBUFFERINSTRUCTIONSPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// ROCDLConfigureBufferInstructionsPass
//===----------------------------------------------------------------------===//
struct ROCDLConfigureBufferInstructionsPass final
    : mlir::iree_compiler::impl::ROCDLConfigureBufferInstructionsPassBase<
          ROCDLConfigureBufferInstructionsPass> {
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Internal functions
//===----------------------------------------------------------------------===//

/// Multiply two numbers, return std::nullopt if an overflow happened.
static std::optional<int64_t> safeMul(int64_t lhs, int64_t rhs) {
  if (lhs > std::numeric_limits<int64_t>::max() / rhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

/// Return the type bit width without using the data layout. TODO: use the data
/// layout.
static unsigned getTypeBitWidth(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type)) {
    return 2 * complexType.getElementType().getIntOrFloatBitWidth();
  }
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return vectorType.getNumElements() *
           getTypeBitWidth(vectorType.getElementType());
  }
  // Over approximate the size of index to not require using the data layout.
  if (type.isIndex()) {
    return 8u;
  }
  // If it's not an int or float return 0.
  return type.isIntOrFloat() ? type.getIntOrFloatBitWidth() : 0;
}

//===----------------------------------------------------------------------===//
// ROCDLConfigureBufferInstructionsPass
//===----------------------------------------------------------------------===//

/// Returns whether the binding op satisfies the requirement of being less than
/// 2GB in size.
static bool
isValidTensorSpan(IREE::HAL::InterfaceBindingSubspanOp op,
                  llvm::function_ref<int64_t(OpFoldResult)> getMax) {
  RankedTensorType tensorTy =
      cast<IREE::TensorExt::DispatchTensorType>(op.getType())
          .asRankedTensorType();
  ArrayRef<int64_t> shape = tensorTy.getShape();

  // Return immediately if the tensor is trivial.
  if (shape.empty()) {
    return true;
  }

  // Compute the max allowed size.
  unsigned byteSize = getTypeBitWidth(tensorTy.getElementType());
  if (byteSize == 0) {
    LDBGS(
        "- Couldn't determine the type size of: " << tensorTy.getElementType());
    return false;
  }
  // Convert from bit-width to bytes.
  byteSize = std::max(byteSize, 8u) / 8u;

  // Get the total size.
  ValueRange dymDims = op.getResultDynamicDims(0);
  int64_t size = 1;
  for (auto [i, rawDim] : llvm::enumerate(shape)) {
    int64_t dim = rawDim;
    if (dim == ShapedType::kDynamic) {
      dim = getMax(dymDims.front());
      // The size is less than 0 if `IntRange` couldn't determine the size.
      if (dim <= 0) {
        LDBGS("- Failed to get the size of the dynamic dimension: " << i);
        return false;
      }
      dymDims = dymDims.drop_front();
    }
    std::optional<int64_t> tmp = safeMul(dim, size);
    if (!tmp) {
      LDBGS("- Overflow detected between the partial total size: "
            << size << ", and shape[" << i << "] = " << dim);
      return false;
    }
    LDBGS("- shape[" << i << "] = " << dim);
    size = *tmp;
  }

  // Get the span in bytes.
  std::optional<int64_t> tmp = safeMul(byteSize, size);
  if (!tmp) {
    LDBGS("- Overflow detected when computing the size of the tensor: "
          << size << ", and " << byteSize);
    return false;
  }
  size = *tmp;
  LDBGS("- The tensor size in bytes is: " << size);
  return static_cast<int64_t>(std::numeric_limits<int32_t>::max()) > size;
}

void ROCDLConfigureBufferInstructionsPass::runOnOperation() {
  if (!clROCDLlEnableBufferInstructions)
    return;
  FunctionOpInterface func = getOperation();

  // Don't perform any actions if the target is not an AMD GPU.
  IREE::GPU::TargetAttr target = getGPUTargetAttr(func);
  if (!target || !target.isAMD())
    return;

  // Configure and run the dataflow analysis.
  DataFlowSolver solver;
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::IntegerRangeAnalysis>();
  solver.load<ROCDL::ThreadUniformAnalysis>();

  if (failed(solver.initializeAndRun(func))) {
    LDBGS(" Dataflow failed, aborting");
    return signalPassFailure();
  }

  // Detect whether a source binding is already annotated and should be skipped.
  auto *gpuDialect = getContext().getLoadedDialect<IREE::GPU::IREEGPUDialect>();
  IREE::GPU::IREEGPUDialect::UseRocdlBufferInstructionsAttrHelper
      annotationHelper = gpuDialect->getUseRocdlBufferInstructionsAttrHelper();

  // Retrieve the max int of a value or -1 if it cannot be determined.
  auto getMax = [&](OpFoldResult ofr) -> int64_t {
    if (auto attr = dyn_cast<Attribute>(ofr)) {
      return cast<IntegerAttr>(attr).getValue().getSExtValue();
    }
    auto *state = solver.lookupState<dataflow::IntegerValueRangeLattice>(
        cast<Value>(ofr));
    if (!state || state->getValue().isUninitialized()) {
      // Return -1 if the state is unknown as we only care for positive values.
      return -1;
    }
    const ConstantIntRanges &range = state->getValue().getValue();
    if (range.umax().getBitWidth() != 0) {
      std::optional<int64_t> m = range.umax().trySExtValue();
      return m ? *m : int64_t{-1};
    }
    if (range.smax().getBitWidth() != 0) {
      std::optional<int64_t> m = range.smax().trySExtValue();
      return m ? *m : int64_t{-1};
    }
    return -1;
  };

  UnitAttr unitAttr = UnitAttr::get(&getContext());
  func.walk([&](IREE::HAL::InterfaceBindingSubspanOp bOp) {
    // Skip if we are not on tensors.
    if (!isa<IREE::TensorExt::DispatchTensorType>(bOp.getType())) {
      return;
    }

    LDBGS(" found binding op: " << bOp);

    // Skip if the op offsets are not thread uniform.
    if (Value offset = bOp.getByteOffset()) {
      auto opAnalysis = solver.lookupState<ROCDL::ThreadUniformLattice>(offset);
      if (!(opAnalysis && opAnalysis->getValue().isUniform())) {
        LDBGS("- Failure, the op offset is not thread uniform");
        return;
      }
    }

    // Check if we can annotate the binding.
    if (!isValidTensorSpan(bOp, getMax)) {
      LDBGS("- Failure, the op byte span is too big");
      return;
    }

    // Annotate the binding.
    LDBGS("- Success, annotating the op");
    annotationHelper.setAttr(bOp, unitAttr);
  });
}
} // namespace mlir::iree_compiler
