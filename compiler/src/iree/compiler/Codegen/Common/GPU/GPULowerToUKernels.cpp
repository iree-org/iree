// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERTOUKERNELSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// Returns a ExecutableObjectAttr carrying the bitcode for the given ukernel.
//
// First tries finding the bitcode in the input `sourceExecutableObjects`, which
// must be an array of ExecutableObjectAttr's and is typically coming from a
// hal.executable.objects array attribute in the source IR, which is the
// mechanism by which source programs may provide their own ukernel bitcode.
//
// If no matching bitcode was found in `sourceExecutableObjects`, this function
// will then search in bitcode files that we have embedded as static data.
static IREE::HAL::ExecutableObjectAttr
getUKernelBitcode(OpBuilder &builder,
                  IREE::HAL::ExecutableTargetAttr execTarget,
                  ArrayAttr sourceExecutableObjects, StringRef ukernelName) {
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(execTarget);
  if (!gpuTarget) {
    return {};
  }
  StringRef gpuArch = gpuTarget.getArch();
  std::string bitcodeFilename = llvm::formatv("{}.{}.bc", ukernelName, gpuArch);

  // Early-return if the source executable.objects already contain an object
  // with the expected file name. This happens with user-provided bitcode in the
  // source IR.
  if (sourceExecutableObjects) {
    for (Attribute a : sourceExecutableObjects) {
      if (auto object = dyn_cast<IREE::HAL::ExecutableObjectAttr>(a)) {
        if (object.getPath() == bitcodeFilename) {
          return object;
        }
      }
    }
  }

  // No user-provided bitcode, so we search our embedded bitcode files in the
  // EmbeddedDataDirectory singleton.
  std::optional<StringRef> bitcode;
  EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
    bitcode = dir.getFile(bitcodeFilename);
  });
  if (!bitcode) {
    return {};
  }
  MLIRContext *context = builder.getContext();
  auto blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(
      ArrayRef<char>(bitcode->data(), bitcode->size()));
  auto bitcodeDenseAttr = DenseI8ResourceElementsAttr::get(
      VectorType::get({static_cast<int64_t>(bitcode->size())},
                      builder.getI8Type()),
      bitcodeFilename, std::move(blob));
  return IREE::HAL::ExecutableObjectAttr::get(
      context, StringAttr::get(context, bitcodeFilename),
      cast<IREE::Util::SerializableAttrInterface>(bitcodeDenseAttr));
}

// Walks parents ops from `op` to return the nearest hal.executable.objects
// array attribute. If the parent hal.executable.variant is reached, its objects
// attribute is returned.
// Adapted from ExecutableTargetAttr::lookup.
static ArrayAttr lookUpExecutableObjects(Operation *op) {
  MLIRContext *context = op->getContext();
  auto attrId = StringAttr::get(context, "hal.executable.objects");
  while (op) {
    // Take directly from the enclosing variant.
    if (auto variantOp = dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
      if (std::optional<ArrayAttr> objects = variantOp.getObjects()) {
        return *objects;
      }
    }
    // Take from op attributes.
    if (auto attr = op->getAttrOfType<ArrayAttr>(attrId)) {
      return attr;
    }
    // Continue walk.
    op = op->getParentOp();
  }
  return {};
}

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
  explicit operator bool() const { return !name.empty(); }
};

/// Returns the function name and attributes to use for a ukernel with given
/// `name` and `suffix` on the target described by `targetAttr`.
static FnNameAndDefAttrs
getFnNameAndDefAttrs(const char *name, std::string &suffix,
                     RewriterBase &rewriter,
                     IREE::HAL::ExecutableTargetAttr targetAttr) {
  FnNameAndDefAttrs result;
  if (isROCMBackend(targetAttr)) {
    result.name = llvm::formatv("iree_uk_amdgpu_{}_{}", name, suffix);
    result.defAttrs.emplace_back(rewriter.getStringAttr("vm.import.module"),
                                 rewriter.getStringAttr("rocm"));
  }
  return result;
}

/// Matches generic that represent argmax and check if
/// we have the ukernel that matches it shape constraint, and types.
/// If we do, then we convert into iree_codegen.ukernel.argmax operation,
/// that is later lowered into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchArgmaxDAGForUKernel(RewriterBase &rewriter, linalg::GenericOp op) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "argmax";
  Value input = op.getDpsInputOperand(0)->get();
  auto inputType = cast<ShapedType>(input.getType());
  Value index = op.getDpsInitOperand(1)->get();
  auto indexType = cast<ShapedType>(index.getType());
  std::string suffix;
  llvm::raw_string_ostream(suffix)
      << inputType.getElementType() << indexType.getElementType();
  FnNameAndDefAttrs fn =
      getFnNameAndDefAttrs(ukernelName, suffix, rewriter, targetAttr);
  if (!fn) {
    return rewriter.notifyMatchFailure(op, "no ukernels on this backend");
  }

  if (!hasUkernel(targetAttr, ukernelName)) {
    return rewriter.notifyMatchFailure(op, "ukernel not enabled");
  }

  // Currently only support argmax where parallel dims are 1.
  // Tiling pipeline is also set to tile all parallel dims to 1, and
  // reduction dim to be size of whole reduction problem. Which allow
  // this constraint to be true for a lot of argmax variances.
  // TODO: Support multi-row or grid-strided argmax ukernel.
  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  SmallVector<unsigned> parallelDims;
  op.getParallelDims(parallelDims);
  int64_t parallelSize = 1;
  for (int64_t dim : parallelDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      return failure();
    }
    parallelSize *= bounds[dim];
  }
  if (parallelSize != 1) {
    return failure();
  }
  auto execTarget = IREE::HAL::ExecutableTargetAttr::lookup(op);
  ArrayAttr sourceExecutableObjects = lookUpExecutableObjects(op);
  IREE::HAL::ExecutableObjectAttr bitcodeObject =
      getUKernelBitcode(rewriter, execTarget, sourceExecutableObjects, fn.name);
  if (!bitcodeObject) {
    return rewriter.notifyMatchFailure(op, "no ukernel bitcode for this op");
  }
  Location loc = op.getLoc();
  // Currently only support 1D reduction, where reduc is on fastest dim.
  // Tiling argmax ukernel is also set to enforce this structure.
  const int kReductionDim = op.getNumLoops() - 1;
  Value reductionDimSize =
      rewriter.create<tensor::DimOp>(loc, input, kReductionDim);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, indexType, fn.name, ValueRange{input}, index,
      ValueRange{reductionDimSize},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(0));
  genericMicroKernelOp->setAttr(
      "hal.executable.objects",
      ArrayAttr::get(rewriter.getContext(), bitcodeObject));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

struct LowerArgmaxToUKernelPattern : OpRewritePattern<linalg::GenericOp> {
  LowerArgmaxToUKernelPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(isArgmaxOp(op))) {
      return failure();
    }
    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchArgmaxDAGForUKernel(rewriter, op);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceAllUsesWith(op.getResults()[1],
                                ukernelOp.value()->getResults());
    return success();
  }
};

struct GPULowerToUKernelsPass final
    : impl::GPULowerToUKernelsPassBase<GPULowerToUKernelsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Enabling a lowering of an op to a microkernel is a trade-off between the
    // potential performance advantage of a microkernel over pure code
    // generation for that op, and the potential benefits of fusions. Indeed,
    // once an op lowered into a microkernel, it will never be fused at any MLIR
    // level. Since microkernels are linked as bitcode, they will still undergo
    // LTO-like optimization in their calling contexts, but we shouldn't expect
    // this to achieve similar results as fusing structured ops.

    // These patterns are unconditionally enabled, because we have strong
    // evidence that it is difficult for codegen to consistently approach
    // microkernels performance, and that consideration overrides the benefit of
    // fusions for these ops.
    patterns.insert<LowerArgmaxToUKernelPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
