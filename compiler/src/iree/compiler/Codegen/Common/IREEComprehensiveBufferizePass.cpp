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

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

namespace mlir {
namespace iree_compiler {

namespace {
class EliminateEmptyTensorsPass
    : public EliminateEmptyTensorsBase<EliminateEmptyTensorsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

/// Pass to convert from tensor based ops to memref based ops.
class IREEComprehensiveBufferizePass
    : public IREEComprehensiveBufferizeBase<IREEComprehensiveBufferizePass> {
public:
  explicit IREEComprehensiveBufferizePass(
      std::optional<BufferizationOptions::AllocationFn> allocationFn =
          std::nullopt,
      std::optional<BufferizationOptions::MemCpyFn> memCpyFn = std::nullopt)
      : allocationFn(allocationFn), memCpyFn(memCpyFn) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                arith::ArithDialect,
                bufferization::BufferizationDialect,
                func::FuncDialect,
                IREE::Flow::FlowDialect,
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
  const std::optional<BufferizationOptions::AllocationFn> allocationFn;
  const std::optional<BufferizationOptions::MemCpyFn> memCpyFn;
};
} // namespace

static bool isaTensor(Type t) { return llvm::isa<TensorType>(t); };

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

static IREEOneShotBufferizationOptions getBufferizationOptions() {
  IREEOneShotBufferizationOptions options;

  // bufferization.to_memref is used to bufferize constants in IREE. IREE has
  // it's own logic to handle constants. We'd like to leave the arith.constant
  // as is and insert bufferization.to_memref to convert the tensor to memref.
  options.opFilter.denyOperation<arith::ConstantOp>();
  options.opFilter.denyOperation<bufferization::ToMemrefOp>();

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
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();

  // Run the convert to destination style patterns.
  {
    RewritePatternSet patterns(context);
    linalg::populateConvertToDestinationStylePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
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
                        const IREEOneShotBufferizationOptions &options) {
  OneShotAnalysisState state(op, options);
  if (failed(analyzeOp(op, state)))
    return failure();
  if (options.testAnalysisOnly)
    return success();
  return bufferization::runOneShotBufferize(op, options);
}

/// Run comprehensive bufferize.
void IREEComprehensiveBufferizePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  IREEOneShotBufferizationOptions options = getBufferizationOptions();
  options.testAnalysisOnly = testAnalysisOnly;
  options.printConflicts = printConflicts;
  options.allocationFn = allocationFn;
  options.memCpyFn = memCpyFn;

  if (failed(runIREEOneShotBufferize(funcOp, options))) {
    return signalPassFailure();
  }

  // Remove redundant args and unused results.
  {
    RewritePatternSet patterns(&getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> createEliminateEmptyTensorsPass() {
  return std::make_unique<EliminateEmptyTensorsPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createIREEComprehensiveBufferizePass(
    std::optional<BufferizationOptions::AllocationFn> allocationFn,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn) {
  if (!allocationFn)
    allocationFn = defaultAllocationFn;
  if (!memCpyFn)
    memCpyFn = defaultMemCpyFn;
  return std::make_unique<IREEComprehensiveBufferizePass>(allocationFn,
                                                          memCpyFn);
}

void addIREEPostBufferizationPasses(OpPassManager &passManager) {
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  // There are redundant memcpy (with linalg.generic form) ops created, which
  // can be deleted by canonicalizer. We have to run it again because the
  // memrefs are unified in CSE pass, so we can truely remove redundant memcpy.
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCleanupBufferAllocViewPass());
}

void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager,
    std::optional<BufferizationOptions::AllocationFn> allocationFn,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn) {
  passManager.addPass(createEliminateEmptyTensorsPass());
  passManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  passManager.addPass(
      createIREEComprehensiveBufferizePass(allocationFn, memCpyFn));
  addIREEPostBufferizationPasses(passManager);
}

} // namespace iree_compiler
} // namespace mlir
