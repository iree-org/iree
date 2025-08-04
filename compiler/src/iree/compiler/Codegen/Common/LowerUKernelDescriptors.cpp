// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenEnums.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-codegen-lower-ukernel-descriptors"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LOWERBITCODEUKERNELSPASS
#define GEN_PASS_DEF_LOWERMEMREFUKERNELSPASS
#define GEN_PASS_DEF_LOWERTENSORUKERNELSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct LowerBitcodeUKernelsPass final
    : impl::LowerBitcodeUKernelsPassBase<LowerBitcodeUKernelsPass> {
  void runOnOperation() override;
};
struct LowerMemrefUKernelsPass final
    : impl::LowerMemrefUKernelsPassBase<LowerMemrefUKernelsPass> {
  void runOnOperation() override;
};
struct LowerTensorUKernelsPass final
    : impl::LowerTensorUKernelsPassBase<LowerTensorUKernelsPass> {
  void runOnOperation() override;
};
} // namespace

template <typename CastOpTy, typename TargetTy>
static void populateCastConversions(TypeConverter &converter) {
  converter.addSourceMaterialization([](OpBuilder &builder, TargetTy resultType,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    if (inputs.size() != 1) {
      return Value();
    }
    Value input = inputs[0];
    auto inputTy = dyn_cast<TargetTy>(input.getType());
    if (!inputTy || !CastOpTy::areCastCompatible(inputTy, resultType)) {
      return Value();
    }
    return builder.create<CastOpTy>(loc, resultType, input).getResult();
  });
  converter.addTargetMaterialization([](OpBuilder &builder, TargetTy resultType,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    if (inputs.size() != 1) {
      return Value();
    }
    Value input = inputs[0];
    auto inputTy = dyn_cast<TargetTy>(input.getType());
    if (!inputTy || !CastOpTy::areCastCompatible(inputTy, resultType)) {
      return Value();
    }
    return builder.create<CastOpTy>(loc, resultType, input).getResult();
  });
}

//===----------------------------------------------------------------------===//
// Conversion/Calling/Inlining Implementations
//===----------------------------------------------------------------------===//

/// Converts an operation to `iree_codegen.ukernel.generic`.
///
/// NOTE: This is primarily an example implementation with inherent limitations.
/// The generic approach used here cannot fulfill the requirements of all
/// ukernel implementations. Real ukernels often need additional
/// context-specific operands (e.g., runtime shapes or algorithm-specific
/// parameters) that cannot be generically inferred from the source operation
/// alone.
static LogicalResult convertToUKernelGeneric(RewriterBase &rewriter,
                                             Operation *op, StringRef name) {
  SmallVector<Value> tensorInputs;
  SmallVector<Value> tensorOutputs;
  SmallVector<Value> otherOperands;
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    // For DPS ops split the operands into inputs and outputs using the
    // interface.
    for (OpOperand &operand : op->getOpOperands()) {
      if (dpsOp.isDpsInput(&operand)) {
        tensorInputs.push_back(operand.get());
      } else if (dpsOp.isDpsInit(&operand)) {
        tensorOutputs.push_back(operand.get());
      } else {
        otherOperands.push_back(operand.get());
      }
    }
  } else {
    for (auto operand : op->getOperands()) {
      // For non-DPS ops, assume all tensor inputs are "inputs" and everything
      // else is one of the "other" operands.
      if (isa<RankedTensorType>(operand.getType())) {
        tensorInputs.push_back(operand);
      } else {
        otherOperands.push_back(operand);
      }
    }
  }
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<IREE::Codegen::UKernelGenericOp>(
      op, op->getResults().getTypes(), name, tensorInputs, tensorOutputs,
      otherOperands, DictionaryAttr(),
      /*strided_outer_dims=*/0);
  return success();
}

/// Replaces the operation `op` with the inlined body of the target function.
/// This performs the necessary type conversions between the operation's
/// operands/results and the function's arguments/return types.
static LogicalResult castAndInline(RewriterBase &rewriter, Operation *op,
                                   FunctionOpInterface targetFunction) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<Value> inputs(op->getOperands());
  ValueRange outputs = op->getResults();

  // Verify that the function argument and result lengths match the inputs and
  // outputs given to this op.
  if (targetFunction.getNumArguments() != inputs.size()) {
    return op->emitError() << "mismatch between number of ukernel arguments "
                           << targetFunction.getNumArguments()
                           << " and number of inputs " << inputs.size();
  }
  if (targetFunction.getNumResults() != outputs.size()) {
    return op->emitError() << "mismatch between number of ukernel results "
                           << targetFunction->getNumResults()
                           << " and number of outputs " << outputs.size();
  }

  // Gather tensor and memref type converters.
  // TODO: Add converters for other types as necessary, e.g. int/float.
  mlir::TypeConverter converter;
  populateCastConversions<tensor::CastOp, RankedTensorType>(converter);
  populateCastConversions<memref::CastOp, MemRefType>(converter);

  rewriter.setInsertionPoint(op);
  for (auto &&[input, type] :
       llvm::zip_equal(inputs, targetFunction.getArgumentTypes())) {
    if (input.getType() != type) {
      Value newInput = converter.materializeSourceConversion(
          rewriter, input.getLoc(), type, input);
      if (!newInput) {
        return op->emitError() << "failed to materialize conversion of "
                               << input << " to type " << type;
      }
      input = newInput;
    }
  }

  SmallVector<Value> replacements;
  // TODO: Support inlining multi-block functions using `scf.execute_region`
  // (needed in case the callsite parent requires a single block region).
  if (!targetFunction.getFunctionBody().hasOneBlock()) {
    return op->emitError()
           << "expected single block function, got "
           << targetFunction.getFunctionBody().getBlocks().size() << " blocks.";
  }

  Region *targetRegion = op->getBlock()->getParent();

  mlir::IRMapping mapper;
  targetFunction.getFunctionBody().cloneInto(targetRegion, mapper);

  Block *body = &targetRegion->getBlocks().back();
  Operation *terminator = body->getTerminator();

  // Inlining the block removes it from the parent region.
  rewriter.inlineBlockBefore(body, op, inputs);
  replacements = terminator->getOperands();
  rewriter.eraseOp(terminator);

  // Cast the call results back to the expected types. If any conversions fail
  // this is a definite failure as the call has been constructed at this point.
  for (auto [output, newOutput] : llvm::zip_equal(outputs, replacements)) {
    Value convertedOutput = newOutput;
    if (output.getType() != newOutput.getType()) {
      convertedOutput = converter.materializeTargetConversion(
          rewriter, output.getLoc(), output.getType(), newOutput);
      if (!convertedOutput) {
        return op->emitError() << "failed to materialize conversion of "
                               << newOutput << " to type " << output.getType();
      }
    }
    rewriter.replaceAllUsesWith(output, convertedOutput);
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementations
//===----------------------------------------------------------------------===//

/// Returns the ops that are marked as ukernels of the provided `kind`.
static SmallVector<std::pair<Operation *, StringRef>>
getOpsToConvert(Operation *root, IREE::Codegen::UKernelArgumentKind kind) {
  SmallVector<std::pair<Operation *, StringRef>> ops;
  root->walk([&](Operation *op) {
    IREE::Codegen::UKernelDescriptorAttr descriptor = getUKernelDescriptor(op);
    if (!descriptor || descriptor.getKind() != kind) {
      return;
    }
    ops.emplace_back(op, descriptor.getUkernelName());
  });
  return ops;
}

static LogicalResult
processUKernelKind(Operation *root, IREE::Codegen::UKernelArgumentKind kind) {
  SmallVector<std::pair<Operation *, StringRef>> opsToConvert =
      getOpsToConvert(root, kind);
  if (opsToConvert.empty()) {
    // Nothing to do if no ops have annotated ukernels.
    return success();
  }

  Operation *annotationSite = nullptr;
  auto targetAttr =
      IREE::HAL::ExecutableTargetAttr::lookup(root, &annotationSite);
  if (!targetAttr) {
    return root->emitError()
           << "no executable target attribute found to resolve ukernels";
  }

  IREE::Codegen::UKernelProviderInterface provider =
      getUKernelProviderFromTarget(targetAttr.getConfiguration());
  if (kind != IREE::Codegen::UKernelArgumentKind::Bitcode && !provider) {
    return root->emitError()
           << "no ukernel provider found to resolve mlir ukernels";
  }

  IRRewriter rewriter(root);
  for (auto [op, name] : opsToConvert) {
    switch (kind) {
    case IREE::Codegen::UKernelArgumentKind::Bitcode: {
      if (failed(convertToUKernelGeneric(rewriter, op, name))) {
        return op->emitOpError()
               << "failed to convert to ukernel.generic with name " << name;
      }
      break;
    }
    case IREE::Codegen::UKernelArgumentKind::Memref:
    case IREE::Codegen::UKernelArgumentKind::Tensor: {
      // Get the function to inline from the ukernel provider.
      FailureOr<Operation *> maybeTargetFunction = provider.getMLIRUKernel(
          name, targetAttr.getConfiguration(), annotationSite);
      if (failed(maybeTargetFunction) || !*maybeTargetFunction) {
        // If not found at the annotation site, look in the first ModuleOp
        // parent as well.
        auto moduleParent = op->getParentOfType<ModuleOp>();
        maybeTargetFunction = provider.getMLIRUKernel(
            name, targetAttr.getConfiguration(), moduleParent);
        if (failed(maybeTargetFunction) || !*maybeTargetFunction) {
          return op->emitOpError()
                 << "failed to retrieve a uKernel with name " << name;
        }
      }
      auto targetFunction = dyn_cast<FunctionOpInterface>(*maybeTargetFunction);
      if (!targetFunction) {
        return op->emitOpError()
               << "failed to retrieve a function-like op for the "
                  "uKernel with name "
               << name;
      }
      if (failed(castAndInline(rewriter, op, targetFunction))) {
        return op->emitOpError()
               << "failed to cast and inline with name " << name;
      }
      break;
    }
    }
  }
  return success();
}

void LowerBitcodeUKernelsPass::runOnOperation() {
  if (failed(processUKernelKind(getOperation(),
                                IREE::Codegen::UKernelArgumentKind::Bitcode))) {
    return signalPassFailure();
  }
}
void LowerMemrefUKernelsPass::runOnOperation() {
  if (failed(processUKernelKind(getOperation(),
                                IREE::Codegen::UKernelArgumentKind::Memref))) {
    return signalPassFailure();
  }
}
void LowerTensorUKernelsPass::runOnOperation() {
  if (failed(processUKernelKind(getOperation(),
                                IREE::Codegen::UKernelArgumentKind::Tensor))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
