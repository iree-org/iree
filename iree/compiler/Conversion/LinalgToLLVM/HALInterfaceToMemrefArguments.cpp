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

#include <memory>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

/// Returns true if the given function contains interface related operations
/// that are used by other ops.
bool containsUsedInterfaceOp(FuncOp funcOp) {
  for (Block& block : funcOp.getBlocks()) {
    for (Operation& op : block) {
      if (!op.getUses().empty() &&
          (isa<IREE::PlaceholderOp>(op) ||
           isa<IREE::HAL::InterfaceLoadConstantOp>(op))) {
        return true;
      }
    }
  }
  return false;
}

/// Returns true if `aOp` has a desciptor (set, binding) pair smaller than
/// `bOp`. Note that this ignores the offset.
bool operator<(IREE::HAL::InterfaceBindingOp aOp,
               IREE::HAL::InterfaceBindingOp bOp) {
  if (aOp.set().getZExtValue() == bOp.set().getZExtValue())
    return aOp.binding().getZExtValue() < bOp.binding().getZExtValue();
  return aOp.set().getZExtValue() < bOp.set().getZExtValue();
}

/// A pattern to process function interface. It replaces interface related ops
/// with function arguments to match LLVM's CodeGen's ABI contract.
///
/// IREE scheduler passes interface ABI information via hal.interface.* ops to
/// all backends. We create iree.placeholder ops to represent buffers behind
/// those hal.interface.* ops. However the LLVM CodeGen uses function parameters
/// and memref descriptors for ABI. So we need to bridge the gap somewhere.
///
/// This pass finds all interface buffers used in the function, sort them
/// according to the descriptor (set, binding) pair, and put unique ones as
/// function parameters in order.
/// Note: This should be kept consistent with LLVM's HAL backend.
struct ProcessFuncInterfacePattern : public OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> Operands,
      ConversionPatternRewriter& rewriter) const override {
    // Only process entry functions.
    if (SymbolTable::getSymbolVisibility(funcOp) !=
        SymbolTable::Visibility::Public)
      return failure();

    FunctionType fnType = funcOp.getType();
    if (fnType.getNumInputs() != 0)
      return rewriter.notifyMatchFailure(
          funcOp, "entry function should not have inputs");

    // Get interface buffers from all the blocks.
    SmallVector<IREE::PlaceholderOp, 8> bufferOps;
    SmallVector<IREE::HAL::InterfaceLoadConstantOp, 8> loadOps;
    for (Block& block : funcOp.getBlocks()) {
      for (Operation& op : block) {
        if (auto phOp = dyn_cast<IREE::PlaceholderOp>(op))
          bufferOps.push_back(phOp);
        if (auto phOp = dyn_cast<IREE::HAL::InterfaceLoadConstantOp>(op)) {
          loadOps.push_back(phOp);
        }
      }
    }

    if (bufferOps.empty()) return failure();

    // A map from buffer ops to their corresponding interface binding ops.
    llvm::DenseMap<Operation*, IREE::HAL::InterfaceBindingOp> bufferBindingMap;
    for (auto bufferOp : bufferOps) {
      auto symbol = SymbolTable::lookupNearestSymbolFrom(
          bufferOp, bufferOp.getAttrOfType<SymbolRefAttr>("binding"));
      bufferBindingMap[bufferOp] = cast<IREE::HAL::InterfaceBindingOp>(symbol);
    }

    // Sort buffers according to their descriptor (set, binding) pair.
    llvm::sort(bufferOps, [&bufferBindingMap](IREE::PlaceholderOp aBuffer,
                                              IREE::PlaceholderOp bBuffer) {
      return bufferBindingMap[aBuffer] < bufferBindingMap[bBuffer];
    });

    // Create a function argument for each of the unique binding pointed by the
    // buffer ops.
    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    // A map from buffer ops to their corresponding function argument indices.
    llvm::DenseMap<Operation*, unsigned> bufferArgMap;
    // A map from binding ops to their corresponding function argument indices.
    llvm::DenseMap<Operation*, unsigned> bindingArgMap;
    unsigned argIndex = 0;
    for (auto bufferOp : bufferOps) {
      auto binding = bufferBindingMap[bufferOp];
      auto it = bindingArgMap.find(binding);
      if (it != bindingArgMap.end()) {
        bufferArgMap[bufferOp] = it->second;
      } else {
        bindingArgMap[binding] = argIndex;
        bufferArgMap[bufferOp] = argIndex;
        signatureConverter.addInputs(bufferOp.getType());
        ++argIndex;
      }
    }
    Type dynamicDimsBufferType =
        MemRefType::get(ShapedType::kDynamicSize, rewriter.getIntegerType(32));
    signatureConverter.addInputs(dynamicDimsBufferType);

    // Create the new function's signature.
    Location loc = funcOp.getLoc();
    auto newFuncOp = rewriter.create<FuncOp>(
        loc, funcOp.getName(),
        rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                 llvm::None),
        ArrayRef<NamedAttribute>());
    newFuncOp.setAttr("llvm.emit_c_interface",
                      mlir::UnitAttr::get(funcOp.getContext()));

    // Move all ops in the old function's region to the new function.
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);

    // Replace all buffer ops' uses with the newly created function arguments
    // and erase them.
    for (auto bufferOp : bufferOps) {
      bufferOp.replaceAllUsesWith(
          newFuncOp.getArgument(bufferArgMap[bufferOp]));

      rewriter.eraseOp(bufferOp);
    }

    // Lower all hal.interface.load.constant ops into std.load
    // from the last buffer holding all dynamic dimensions with the proper
    // offset.
    Type indexType = rewriter.getIndexType();
    auto builder = OpBuilder::atBlockBegin(&(newFuncOp.getBlocks().front()));
    auto newLoc = newFuncOp.front().front().getLoc();
    for (auto loadOp : loadOps) {
      SmallVector<Value, 1> indices;
      Value constantOffset = builder.create<ConstantOp>(
          newLoc, indexType,
          rewriter.getIntegerAttr(indexType, loadOp.offset().getZExtValue()));
      indices.push_back(constantOffset);
      Value loadDim = builder.create<LoadOp>(
          newLoc, newFuncOp.getArgument(newFuncOp.getNumArguments() - 1),
          indices);
      Value loadDimIndex =
          builder.create<IndexCastOp>(newLoc, loadDim, indexType);
      loadOp.replaceAllUsesWith(loadDimIndex);
      rewriter.eraseOp(loadOp);
    }
    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct RemoveInterfaceOpPattern
    : public OpRewritePattern<IREE::HAL::InterfaceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::InterfaceOp interfaceOp,
                                PatternRewriter& rewriter) const override {
    rewriter.eraseOp(interfaceOp);
    return success();
  }
};

/// Converting from Linalg to LLVM needs to run on a module and since it
/// applies a full conversion, make a module with jst the impl function.
struct HALInterfaceToMemrefArgumentsPass
    : PassWrapper<HALInterfaceToMemrefArgumentsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext& context = getContext();

    OwningRewritePatternList patterns;
    patterns.insert<ProcessFuncInterfacePattern>(&context);
    patterns.insert<RemoveInterfaceOpPattern>(&context);

    ConversionTarget target(context);
    // Convert the interface related ops away.
    target.addDynamicallyLegalOp<FuncOp>(
        [](FuncOp funcOp) { return !containsUsedInterfaceOp(funcOp); });
    target.addIllegalOp<IREE::PlaceholderOp>();
    target.addIllegalDialect<IREE::HAL::HALDialect>();
    // Allow the rest.
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyFullConversion(getOperation(), target, patterns)))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createHALInterfaceToMemrefArgumentsPass() {
  return std::make_unique<HALInterfaceToMemrefArgumentsPass>();
}

static PassRegistration<HALInterfaceToMemrefArgumentsPass> pass(
    "iree-codegen-hal-interface-to-memref-arguments-pass",
    "Convert a function with HAL bindings interface to memref arguments",
    [] { return std::make_unique<HALInterfaceToMemrefArgumentsPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
