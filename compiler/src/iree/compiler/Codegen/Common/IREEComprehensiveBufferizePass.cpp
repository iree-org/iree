// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREEComprehensiveBufferizePass.cpp - -------------------------------===//
//
// Wrapper pass to use MLIR's One-Shot Bufferize pass.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-linalg-bufferize"

using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::OneShotAnalysisState;
using mlir::bufferization::OneShotBufferizationOptions;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ELIMINATEEMPTYTENSORSPASS
#define GEN_PASS_DEF_IREEBUFFERIZECONSTANTSPASS
#define GEN_PASS_DEF_IREECOMPREHENSIVEBUFFERIZEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

// Default allocation functions.
static FailureOr<Value> defaultAllocationFn(OpBuilder &builder, Location loc,
                                            MemRefType allocationType,
                                            ValueRange dynamicSizes,
                                            unsigned int alignment) {
  MemRefType type = allocationType;
  if (auto storage = type.getMemorySpace()) {
    // We cannot allocate to generate a resultant MemRef type with descriptor
    // type memory space; that's runtime allocations. So erase and fallback to
    // the default 0 memory space. It is fine given this is just the default
    // allocator; backends are expected to control by themselves.
    if (llvm::isa<IREE::HAL::DescriptorTypeAttr>(storage))
      type = MemRefType::get(type.getShape(), type.getElementType(),
                             type.getLayout());
  }
  return builder.create<memref::AllocOp>(loc, type, dynamicSizes).getResult();
}
static LogicalResult defaultMemCpyFn(OpBuilder &builder, Location loc,
                                     Value from, Value to) {
  Operation *copyOp = createLinalgCopyOp(builder, loc, from, to);
  return success(static_cast<bool>(copyOp));
}

namespace {
class EliminateEmptyTensorsPass final
    : public impl::EliminateEmptyTensorsPassBase<EliminateEmptyTensorsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

/// Pass to convert from tensor based ops to memref based ops.
class IREEComprehensiveBufferizePass final
    : public impl::IREEComprehensiveBufferizePassBase<
          IREEComprehensiveBufferizePass> {
public:
  using impl::IREEComprehensiveBufferizePassBase<
      IREEComprehensiveBufferizePass>::IREEComprehensiveBufferizePassBase;
  explicit IREEComprehensiveBufferizePass(
      BufferizationOptions::AllocationFn allocationFn,
      BufferizationOptions::MemCpyFn memCpyFn)
      : allocationFn(allocationFn), memCpyFn(memCpyFn) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                amdgpu::AMDGPUDialect,
                arith::ArithDialect,
                bufferization::BufferizationDialect,
                func::FuncDialect,
                gpu::GPUDialect,
                IREE::LinalgExt::IREELinalgExtDialect,
                IREE::Util::UtilDialect,
                linalg::LinalgDialect,
                memref::MemRefDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                vector::VectorDialect>();
    // clang-format on
  }

  void runOnOperation() override;

private:
  const BufferizationOptions::AllocationFn allocationFn = defaultAllocationFn;
  const BufferizationOptions::MemCpyFn memCpyFn = defaultMemCpyFn;
};

/// Pass to convert from tensor based constants to memref.
class IREEBufferizeConstantsPass final
    : public impl::IREEBufferizeConstantsPassBase<IREEBufferizeConstantsPass> {
public:
  using impl::IREEBufferizeConstantsPassBase<
      IREEBufferizeConstantsPass>::IREEBufferizeConstantsPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override;
};
} // namespace

static IREEOneShotBufferizationOptions getBufferizationOptions() {
  IREEOneShotBufferizationOptions options;

  // bufferization.to_buffer is used to bufferize constants in IREE. IREE has
  // it's own logic to handle constants. We'd like to leave the arith.constant
  // as is and insert bufferization.to_buffer to convert the tensor to memref.
  options.opFilter.denyOperation<arith::ConstantOp>();
  options.opFilter.denyOperation<bufferization::ToBufferOp>();

  // This type converter converts tensor types to memref types when no exact
  // memref type can be inferred from the context.
  options.unknownTypeConverterFn = [](Value value, Attribute memorySpace,
                                      const BufferizationOptions &options) {
    auto tensorType = llvm::cast<TensorType>(value.getType());

    // Special rule for ConstantOps: These always lower to some memref with a
    // static identity layout.
    if (value.getDefiningOp<arith::ConstantOp>())
      return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                  memorySpace);

    // Default case: Fully dynamic layout map for best compatibility.
    return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                              memorySpace);
  };

  return options;
}

LogicalResult
eliminateEmptyTensors(RewriterBase &rewriter, Operation *op,
                      const OneShotBufferizationOptions &options) {
  // Analyze IR.
  OneShotAnalysisState state(op, options);
  if (failed(analyzeOp(op, state)))
    return failure();

  // Rewrite tensor.empty ops that are anchored on specific ops.
  if (failed(bufferization::eliminateEmptyTensors(rewriter, op, state)))
    return failure();

  return success();
}

void EliminateEmptyTensorsPass::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();

  OpBuilder b(context);
  SmallVector<tensor::EmptyOp> emptyOps;
  funcOp.walk([&](tensor::EmptyOp emptyOp) { emptyOps.push_back(emptyOp); });
  if (llvm::any_of(emptyOps, [&](tensor::EmptyOp emptyOp) {
        return failed(duplicateTensorEmptyOps(b, emptyOp));
      })) {
    return signalPassFailure();
  }

  // Run the convert to destination style patterns.
  {
    RewritePatternSet patterns(context);
    linalg::populateConvertToDestinationStylePatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp->emitOpError("Failed in conversion to destination style patterns");
      return signalPassFailure();
    }
  }

  IRRewriter rewriter(funcOp->getContext());
  auto bufferizationOptions = getBufferizationOptions();
  OneShotAnalysisState state(funcOp, bufferizationOptions);
  // Analyze IR.
  if (failed(analyzeOp(funcOp, state)))
    return signalPassFailure();
  // Eliminate empty tensors.
  if (failed(bufferization::eliminateEmptyTensors(rewriter, funcOp, state)))
    return signalPassFailure();
}

// The following is copied from bufferization::runOneShotBufferize with
// modifications.
LogicalResult
runIREEOneShotBufferize(Operation *op,
                        const IREEOneShotBufferizationOptions &options,
                        bufferization::BufferizationState &state) {
  OneShotAnalysisState analyzeState(op, options);
  if (failed(analyzeOp(op, analyzeState)))
    return failure();
  if (options.testAnalysisOnly)
    return success();
  return bufferization::runOneShotBufferize(op, options, state);
}

/// Run comprehensive bufferize.
void IREEComprehensiveBufferizePass::runOnOperation() {
  auto funcOp = getOperation();
  IREEOneShotBufferizationOptions options = getBufferizationOptions();
  options.testAnalysisOnly = testAnalysisOnly;
  options.printConflicts = printConflicts;
  options.allocationFn = allocationFn;
  options.memCpyFn = memCpyFn;
  // Turning off checkParallelRegions assumes that we are not relying too much
  // on bufferization being conservative. If we are, then this could cause race
  // conditions. Turning this option off could be a good step in diagnosing
  // data races on GPU.
  options.checkParallelRegions = false;

  bufferization::BufferizationState bufferizationState;
  if (failed(runIREEOneShotBufferize(funcOp, options, bufferizationState))) {
    return signalPassFailure();
  }

  // Remove redundant args and unused results.
  {
    RewritePatternSet patterns(&getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

void IREEBufferizeConstantsPass::runOnOperation() {
  mlir::bufferization::OneShotBufferizationOptions opt;
  opt.copyBeforeWrite = true;
  opt.opFilter.allowOperation(arith::ConstantOp::getOperationName());
  bufferization::BufferizationState bufferizationState;
  if (failed(mlir::bufferization::runOneShotBufferize(
          getOperation(), opt, bufferizationState,
          /*statistics=*/nullptr))) {
    signalPassFailure();
    return;
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createIREEComprehensiveBufferizePass(
    std::optional<BufferizationOptions::AllocationFn> allocationFn,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn) {
  if (!allocationFn)
    allocationFn = defaultAllocationFn;
  if (!memCpyFn)
    memCpyFn = defaultMemCpyFn;
  return std::make_unique<IREEComprehensiveBufferizePass>(allocationFn.value(),
                                                          memCpyFn.value());
}

void addIREEPostBufferizationPasses(OpPassManager &funcPassManager,
                                    bool injectAssumeAlignmentOp) {
  if (injectAssumeAlignmentOp) {
    funcPassManager.addPass(createIREEInjectAssumeAlignmentPass());
  }
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  // There are redundant memcpy (with linalg.generic form) ops created, which
  // can be deleted by canonicalizer. We have to run it again because the
  // memrefs are unified in CSE pass, so we can truely remove redundant memcpy.
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCleanupBufferAllocViewPass());
}

void addIREEComprehensiveBufferizePasses(
    OpPassManager &funcPassManager,
    std::optional<BufferizationOptions::AllocationFn> allocationFn,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn,
    bool injectAssumeAlignmentOp) {
  funcPassManager.addPass(createEliminateEmptyTensorsPass());
  funcPassManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  funcPassManager.addPass(
      createIREEComprehensiveBufferizePass(allocationFn, memCpyFn));
  addIREEPostBufferizationPasses(funcPassManager, injectAssumeAlignmentOp);
}

} // namespace mlir::iree_compiler
