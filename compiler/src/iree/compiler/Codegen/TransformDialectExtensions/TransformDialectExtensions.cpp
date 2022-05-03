// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TransformDialectExtensions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformOpInterface.h"
#include "iree-dialects/Transforms/Functional.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE;

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
// TODO: register the bufferization behavior in a target-specific way.
//===---------------------------------------------------------------------===//

// Default allocation function to use with IREEs bufferization.
static Value cpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
}

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  return success();
}

static LogicalResult cpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced as
  // an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  //   builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

//===---------------------------------------------------------------------===//
// IREE-specific transformations defined outside of iree_linalg_transform.
//===---------------------------------------------------------------------===//

// Note: with the recent TypeID changes, hiding these classes inside an
// anonymous namespace would require specific `MLIR_DECLARE_EXPLICIT_TYPE_ID`
// for each class.

// namespace {

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is enough.
class IREEBufferizeOp
    : public Op<IREEBufferizeOp,
                linalg::transform::TransformOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("iree_linalg_transform.iree_bufferize");
  }

  Value target() { return nullptr; }

  LogicalResult apply(linalg::transform::TransformResults &results,
                      linalg::transform::TransformState &state) {
    PassManager pm(getContext());
    // Bufferize the dispatch.
    using mlir::bufferization::BufferizationOptions;
    BufferizationOptions::AllocationFn allocationFn =
        cpuComprehensiveBufferizeAllocationFn;
    BufferizationOptions::DeallocationFn deallocationFn =
        cpuComprehensiveBufferizeDeallocationFn;
    BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
    mlir::iree_compiler::addIREEComprehensiveBufferizePasses(
        pm, allocationFn, deallocationFn, memcpyFn);
    WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
      if (failed(pm.run(moduleOp))) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    parser.parseOptionalAttrDict(state.attributes);
    return success();
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }
};

// TODO: Move to tablegen. Until this stabilizes upstream, simple C++ is enough.
class IREESetNumWorkgroupToOneOp
    : public Op<IREESetNumWorkgroupToOneOp,
                linalg::transform::TransformOpInterface::Trait> {
 public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral(
        "iree_linalg_transform.iree_set_num_workgroups_to_one");
  }

  Value target() { return nullptr; }

  LogicalResult apply(linalg::transform::TransformResults &results,
                      linalg::transform::TransformState &state) {
    auto variantOp = dyn_cast<HAL::ExecutableVariantOp>(state.getTopLevel());
    if (!variantOp) return failure();
    return iree_compiler::setNumWorkgroupsImpl(variantOp, {});
  }

  // let assemblyFormat = "attr-dict";
  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    parser.parseOptionalAttrDict(state.attributes);
    return success();
  }

  // let assemblyFormat = "attr-dict";
  void print(OpAsmPrinter &printer) {
    printer.printOptionalAttrDict((*this)->getAttrs());
  }
};

/// Test extension of the Transform dialect. Registers additional ops and
/// declares PDL as dependent dialect since the additional ops are using PDL
/// types for operands and results.
class LinalgTransformDialectExtension
    : public mlir::linalg::transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
 public:
  LinalgTransformDialectExtension() {
    declareDependentDialect<pdl::PDLDialect>();
    registerTransformOp<IREEBufferizeOp>();
    registerTransformOp<IREESetNumWorkgroupToOneOp>();
    // TODO: hook up to Tablegen.
    //     registerTransformOps<
    // #define GET_OP_LIST
    // #include "LinalgTransformDialectExtension.cpp.inc"
    //         >();
  }
};

// } // namespace anonymous

void mlir::iree_compiler::registerLinalgTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
