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

#include "iree/compiler/Translation/XLAToLinalg/LinalgTensorToBuffer.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"

namespace mlir {
namespace iree_compiler {
namespace {

/// Convert from a linalg.generic on tensors to linalg.generic on buffers. In
/// IREE it is expected that each dispatch region will become a single
/// linalg.generic op on tensors (after XLA-HLO -> Linalg conversion and
/// fusion). So linalg.generic on tensors can be changed to linalg.buffer on
/// memrefs without doing buffer allocation, and by using the arguments to the
/// dispatch function as arguments to the linalg.generic op. This patterns only
/// checks for cases where the operands are results of iree.load_input, and the
/// result is an argument to iree.store_output.  Writing this as a dialect
/// conversion pattern (even though going from linalg -> linalg) to access the
/// signature conversion methods.
struct LinalgTensorToBufferConverter
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;
  PatternMatchResult matchAndRewrite(
      linalg::GenericOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

/// Remove IREE::StoreOutputOp operations.
struct RemoveDeadStorePattern : OpConversionPattern<IREE::StoreOutputOp> {
  using OpConversionPattern<IREE::StoreOutputOp>::OpConversionPattern;
  PatternMatchResult matchAndRewrite(
      IREE::StoreOutputOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const {
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

/// Replace iree.return with std.return operation.
struct IREEReturnOpLowering : OpConversionPattern<IREE::ReturnOp> {
  using OpConversionPattern<IREE::ReturnOp>::OpConversionPattern;
  PatternMatchResult matchAndRewrite(
      IREE::ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return matchSuccess();
  }
};

}  // namespace

PatternMatchResult LinalgTensorToBufferConverter::matchAndRewrite(
    linalg::GenericOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // TODO(ravishankarm): Find a way to write this using Matchers, but need to
  // figure out how to match operations with variadic operands.
  SmallVector<Value, 2> memrefArgs;
  for (auto arg : op.getOperands()) {
    if (!arg.getType().isa<RankedTensorType>()) {
      return matchFailure();
    }
    auto definingOp = dyn_cast_or_null<IREE::LoadInputOp>(arg.getDefiningOp());
    if (!definingOp) {
      return matchFailure();
    }
    memrefArgs.push_back(definingOp.getOperand());
  }
  // For result, check that there is a single use in an iree::store_output op.
  for (auto result : op.getResults()) {
    if (!result.hasOneUse()) {
      return matchFailure();
    }
    auto resultUser = dyn_cast<IREE::StoreOutputOp>(*result.user_begin());
    memrefArgs.push_back(resultUser.dst());
  }

  // Create a new op with the same traits as the original generic op, but with
  // memrefs.
  // TODO(ravishankarm): Figure out how to do this inplace.
  auto linalgBufferOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(), ArrayRef<Type>(), memrefArgs, op.args_in(), op.args_out(),
      op.indexing_maps(), op.iterator_types(),
      /*doc=*/nullptr,
      /*fun=*/nullptr,
      /*library_call=*/nullptr);
  // Move the region from the replaced op into the new op.
  unsigned numTensorOperands = op.getNumOperands();
  auto &region = linalgBufferOp.region();
  region.takeBody(op.region());
  // Need to convert the signature to take extra arguments for the return type.
  TypeConverter::SignatureConversion signatureConverter(numTensorOperands);
  for (auto arg : llvm::enumerate(memrefArgs)) {
    if (arg.index() < numTensorOperands) {
      signatureConverter.addInputs(
          arg.index(),
          arg.value().getType().cast<MemRefType>().getElementType());
    } else {
      signatureConverter.addInputs(
          arg.value().getType().cast<MemRefType>().getElementType());
    }
  }
  rewriter.applySignatureConversion(&region, signatureConverter);
  rewriter.eraseOp(op);
  return matchSuccess();
}

void populateLinalgTensorToBufferConversionPattern(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<LinalgTensorToBufferConverter, RemoveDeadStorePattern,
                  IREEReturnOpLowering>(context);
}

struct LinalgTensorToBufferConversionPass
    : public FunctionPass<LinalgTensorToBufferConversionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
    target.addLegalOp<IREE::LoadInputOp>();
    target.addLegalOp<FuncOp>();
    target.addDynamicallyLegalOp<linalg::GenericOp>([&](linalg::GenericOp op) {
      return llvm::all_of(op.getOperands(),
                          [](Value v) -> bool {
                            return v.getType().isa<MemRefType>();
                          }) &&
             op.getResults().empty();
    });

    populateLinalgTensorToBufferConversionPattern(context, patterns);
    if (failed(applyFullConversion(getFunction(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createLinalgTensorToBufferConversionPass() {
  return std::make_unique<LinalgTensorToBufferConversionPass>();
}

static PassRegistration<LinalgTensorToBufferConversionPass> pass(
    "iree-linalg-tensor-to-buffer",
    "Convert linalg.generic op on tensors to linalg.generic op on memrefs for "
    "IREE dispatch functions");

}  // namespace iree_compiler
}  // namespace mlir
