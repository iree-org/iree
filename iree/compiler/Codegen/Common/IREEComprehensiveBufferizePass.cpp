// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREEComprehensiveBufferizePass.cpp - -------------------------===//
//
// Wrapper pass to use MLIRs ComprehensiveBufferization pass.
//
//===----------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/BufferizationAnalysis.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-linalg-bufferize"

using mlir::bufferization::AnalysisBufferizationOptions;
using mlir::bufferization::BufferizationOptions;

namespace mlir {
namespace iree_compiler {

namespace {

/// Pass to convert from tensor based ops to memref based ops.
class IREEComprehensiveBufferizePass
    : public IREEComprehensiveBufferizeBase<IREEComprehensiveBufferizePass> {
 public:
  explicit IREEComprehensiveBufferizePass(
      Optional<BufferizationOptions::AllocationFn> allocationFn = None,
      Optional<BufferizationOptions::DeallocationFn> deallocationFn = None,
      Optional<BufferizationOptions::MemCpyFn> memCpyFn = None)
      : allocationFn(allocationFn),
        deallocationFn(deallocationFn),
        memCpyFn(memCpyFn) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithmeticDialect, IREE::Util::UtilDialect,
                linalg::LinalgDialect, memref::MemRefDialect, scf::SCFDialect,
                func::FuncDialect, tensor::TensorDialect, vector::VectorDialect,
                AffineDialect, IREE::Flow::FlowDialect,
                bufferization::BufferizationDialect>();
  }

  void runOnOperation() override;

 private:
  const Optional<BufferizationOptions::AllocationFn> allocationFn;
  const Optional<BufferizationOptions::DeallocationFn> deallocationFn;
  const Optional<BufferizationOptions::MemCpyFn> memCpyFn;
};
}  // namespace

static bool isaTensor(Type t) { return t.isa<TensorType>(); };

/// Run comprehensive bufferize.
void IREEComprehensiveBufferizePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  AnalysisBufferizationOptions options;
  options.allocationFn = allocationFn;
  options.deallocationFn = deallocationFn;
  options.memCpyFn = memCpyFn;
  options.testAnalysisOnly = testAnalysisOnly;
  options.printConflicts = printConflicts;
  options.alwaysAliasingWithDest = false;
  addPostAnalysisTransformations(options);

  if (failed(bufferization::runOneShotBufferize(moduleOp, options))) {
    return signalPassFailure();
  }
}

// Default allocation functions.
static FailureOr<Value> defaultAllocationFn(OpBuilder &builder, Location loc,
                                            MemRefType allocationType,
                                            ValueRange dynamicSizes,
                                            unsigned int alignment) {
  return builder.create<memref::AllocOp>(loc, allocationType, dynamicSizes)
      .getResult();
}
static LogicalResult defaultDeallocationFn(OpBuilder &builder, Location loc,
                                           Value allocation) {
  builder.create<memref::DeallocOp>(loc, allocation);
  return success();
}
static LogicalResult defaultMemCpyFn(OpBuilder &builder, Location loc,
                                     Value from, Value to) {
  createLinalgCopyOp(builder, loc, from, to);
  return success();
}

std::unique_ptr<OperationPass<ModuleOp>> createIREEComprehensiveBufferizePass(
    Optional<BufferizationOptions::AllocationFn> allocationFn,
    Optional<BufferizationOptions::DeallocationFn> deallocationFn,
    Optional<BufferizationOptions::MemCpyFn> memCpyFn) {
  if (!allocationFn) allocationFn = defaultAllocationFn;
  if (!deallocationFn) deallocationFn = defaultDeallocationFn;
  if (!memCpyFn) memCpyFn = defaultMemCpyFn;
  return std::make_unique<IREEComprehensiveBufferizePass>(
      allocationFn, deallocationFn, memCpyFn);
}

void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager,
    Optional<BufferizationOptions::AllocationFn> allocationFn,
    Optional<BufferizationOptions::DeallocationFn> deallocationFn,
    Optional<BufferizationOptions::MemCpyFn> memCpyFn) {
  passManager.addPass(createIREEComprehensiveBufferizePass(
      allocationFn, deallocationFn, memCpyFn));
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  // There are redundant memcpy (with linalg.generic form) ops created, which
  // can be deleted by canonicalizer. We have to run it again because the
  // memrefs are unified in CSE pass, so we can truely remove redundant memcpy.
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCleanupBufferAllocViewPass());
}

}  // namespace iree_compiler
}  // namespace mlir
