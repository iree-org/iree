// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREEComprehensiveBufferizePass.cpp.cpp - -------------------------===//
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
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-linalg-bufferize"

namespace mlir {
namespace iree_compiler {

using linalg::comprehensive_bufferize::BufferizableOpInterface;
using linalg::comprehensive_bufferize::BufferizationAliasInfo;
using linalg::comprehensive_bufferize::BufferizationState;

namespace {

/// Pass to convert from tensor based ops to memref based ops.
class IREEComprehensiveBufferizePass
    : public IREEComprehensiveBufferizeBase<IREEComprehensiveBufferizePass> {
 public:
  explicit IREEComprehensiveBufferizePass(
      std::unique_ptr<linalg::comprehensive_bufferize::AllocationCallbacks>
          allocationFn)
      : allocationFn(std::move(allocationFn)) {}

  IREEComprehensiveBufferizePass(const IREEComprehensiveBufferizePass &other) {
    allocationFn =
        std::make_unique<linalg::comprehensive_bufferize::AllocationCallbacks>(
            other.allocationFn->allocationFn,
            other.allocationFn->deallocationFn, other.allocationFn->memCpyFn);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithmeticDialect, IREE::Util::UtilDialect,
                linalg::LinalgDialect, memref::MemRefDialect, scf::SCFDialect,
                StandardOpsDialect, tensor::TensorDialect,
                vector::VectorDialect, AffineDialect, IREE::Flow::FlowDialect,
                bufferization::BufferizationDialect>();
  }

  void runOnOperation() override;

 private:
  std::unique_ptr<linalg::comprehensive_bufferize::AllocationCallbacks>
      allocationFn;
};
}  // namespace

static bool isaTensor(Type t) { return t.isa<TensorType>(); };

/// Run comprehensive bufferize.
void IREEComprehensiveBufferizePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  auto options =
      std::make_unique<linalg::comprehensive_bufferize::BufferizationOptions>();
  options->allocationFns =
      std::make_unique<linalg::comprehensive_bufferize::AllocationCallbacks>(
          allocationFn->allocationFn, allocationFn->deallocationFn,
          allocationFn->memCpyFn);
  options->testAnalysisOnly = false;
  addPostAnalysisTransformations(*options);

  if (failed(runComprehensiveBufferize(moduleOp, std::move(options)))) {
    return signalPassFailure();
  }
}

// Default allocation functions.
static Optional<Value> defaultAllocationFn(OpBuilder &builder, Location loc,
                                           MemRefType allocationType,
                                           ArrayRef<Value> dynamicSizes) {
  return builder.create<memref::AllocOp>(loc, allocationType, dynamicSizes)
      .getResult();
}
static void defaultDeallocationFn(OpBuilder &builder, Location loc,
                                  Value allocation) {
  builder.create<memref::DeallocOp>(loc, allocation);
}
static void defaultMemCpyFn(OpBuilder &builder, Location loc, Value from,
                            Value to) {
  builder.create<linalg::CopyOp>(loc, from, to);
}

std::unique_ptr<OperationPass<ModuleOp>> createIREEComprehensiveBufferizePass(
    std::unique_ptr<linalg::comprehensive_bufferize::AllocationCallbacks>
        allocationFns) {
  if (!allocationFns) {
    allocationFns =
        std::make_unique<linalg::comprehensive_bufferize::AllocationCallbacks>(
            defaultAllocationFn, defaultDeallocationFn, defaultMemCpyFn);
  }
  return std::make_unique<IREEComprehensiveBufferizePass>(
      std::move(allocationFns));
}

void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager,
    std::unique_ptr<linalg::comprehensive_bufferize::AllocationCallbacks>
        allocationFns) {
  passManager.addNestedPass<FuncOp>(
      createConvertToDestinationPassingStylePass());
  passManager.addPass(
      createIREEComprehensiveBufferizePass(std::move(allocationFns)));
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCleanupBufferAllocViewPass());
}

}  // namespace iree_compiler
}  // namespace mlir
