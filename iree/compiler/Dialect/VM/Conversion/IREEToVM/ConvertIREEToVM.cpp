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

#include "iree/compiler/Dialect/VM/Conversion/IREEToVM/ConvertIREEToVM.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// iree.byte_buffer.*
//===----------------------------------------------------------------------===//

class ByteBufferConstantOpConversion
    : public OpConversionPattern<IREE::ByteBufferConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::ByteBufferConstantOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::RodataInlineOp>(
        op, IREE::VM::RefType::get(op.getType()), op.value());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

class UnreachableOpConversion
    : public OpConversionPattern<IREE::UnreachableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::UnreachableOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::FailOp>(
        srcOp,
        rewriter.createOrFold<mlir::ConstantIntOp>(
            srcOp.getLoc(), static_cast<int32_t>(IREE::StatusCode::Unknown),
            32),
        "unreachable location reached");
    return success();
  }
};

}  // namespace

void populateIREEToVMPatterns(MLIRContext *context,
                              OwningRewritePatternList &patterns) {
  patterns.insert<ByteBufferConstantOpConversion>(context);
  patterns.insert<UnreachableOpConversion>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
