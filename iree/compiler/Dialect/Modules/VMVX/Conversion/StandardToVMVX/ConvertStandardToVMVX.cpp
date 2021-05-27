// Copyright 2021 Google LLC
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

#include "iree/compiler/Dialect/Modules/VMVX/Conversion/StandardToVMVX/ConvertStandardToVMVX.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Pattern to lower operations that become a no-ops at this level.
template <typename OpTy>
struct FoldAsNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands);
    return success();
  }
};

/// Removes unrealized_conversion_cast ops introduced during progressive
/// lowering when possible.
struct RemoveIdentityConversionCast final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 1 && op->getNumResults() == 1 &&
        operands.front().getType() == op->getResultTypes().front()) {
      rewriter.replaceOp(op, operands);
      return success();
    }

    return failure();
  }
};

}  // namespace

void populateStandardToVMVXPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns,
                                    TypeConverter &typeConverter) {
  // We type/shape erase memrefs as we lower so there is no need for reshapes.
  patterns.insert<FoldAsNoOp<linalg::ReshapeOp>>(typeConverter, context);

  patterns.insert<RemoveIdentityConversionCast>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
