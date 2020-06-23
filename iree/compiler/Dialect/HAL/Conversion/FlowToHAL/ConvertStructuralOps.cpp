// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class FuncOpSignatureConversion : public OpConversionPattern<mlir::FuncOp> {
 public:
  FuncOpSignatureConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx), converter(converter) {}

  LogicalResult matchAndRewrite(
      mlir::FuncOp funcOp, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Convert the input signature types.
    // TODO(benvanik): dynamic shapes by passing in tensor dynamic dims.
    auto originalType = funcOp.getType();
    TypeConverter::SignatureConversion newSignature(
        originalType.getNumInputs());
    for (auto argType : llvm::enumerate(originalType.getInputs())) {
      if (failed(converter.convertSignatureArg(argType.index(), argType.value(),
                                               newSignature))) {
        return failure();
      }
    }
    SmallVector<Type, 4> newResultTypes;
    if (failed(converter.convertTypes(originalType.getResults(),
                                      newResultTypes))) {
      return failure();
    }

    // Replace function.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    newFuncOp.getBlocks().clear();
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    newFuncOp.setType(rewriter.getFunctionType(newSignature.getConvertedTypes(),
                                               newResultTypes));
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), converter,
                                           &newSignature)))
      return failure();

    rewriter.eraseOp(funcOp);
    return success();
  }

 private:
  TypeConverter &converter;
};

}  // namespace

void populateFlowStructuralToHALPatterns(MLIRContext *context,
                                         OwningRewritePatternList &patterns,
                                         TypeConverter &converter) {
  patterns.insert<FuncOpSignatureConversion>(context, converter);
}

}  // namespace iree_compiler
}  // namespace mlir
