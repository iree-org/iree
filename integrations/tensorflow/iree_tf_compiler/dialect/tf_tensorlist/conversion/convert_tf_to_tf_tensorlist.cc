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

#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace iree_integrations {
namespace tf_tensorlist {

namespace {
TypeAttr GetVariantElementTypeAttr(Type type) {
  if (auto variantTy = type.dyn_cast<TF::VariantType>()) {
    return GetVariantElementTypeAttr(variantTy.getSubtypes().front());
  }

  if (auto shapedTy = type.dyn_cast<ShapedType>()) {
    return GetVariantElementTypeAttr(shapedTy.getElementType());
  }

  return TypeAttr::get(type);
}

}  // namespace

#include "iree_tf_compiler/dialect/tf_tensorlist/conversion/convert_tf_to_tf_tensorlist.inc"

class ConvertTFToTFTensorListPass
    : public PassWrapper<ConvertTFToTFTensorListPass, OperationPass<FuncOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::TF::TensorFlowDialect, TFTensorListDialect,
                    StandardOpsDialect>();
  }
  void runOnOperation() override;
};

bool isTfVariant(Type type) {
  if (auto tensorType = type.dyn_cast<TensorType>()) {
    return tensorType.getElementType().isa<TF::VariantType>();
  }
  return false;
}

namespace {
class ConvertTfTensorlistConcatV2
    : public OpRewritePattern<TF::TensorListConcatV2Op> {
 public:
  using OpRewritePattern<TF::TensorListConcatV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TensorListConcatV2Op op,
                                PatternRewriter &rewriter) const override {
    Value tensor_list = op.input_handle();
    Value out_tensor = op.tensor();
    Value out_lengths = op.lengths();

    Value concat = rewriter.create<tf_tensorlist::Concat>(
        op.getLoc(), out_tensor.getType(), tensor_list);
    Value dim0Lengths = rewriter.create<tf_tensorlist::GetDim0>(
        op.getLoc(), out_lengths.getType(), tensor_list);

    rewriter.replaceOp(op, {concat, dim0Lengths});
    return success();
  }
};
}  // namespace

void ConvertTFToTFTensorListPass::runOnOperation() {
  auto func = getOperation();

  // The conversion happens in 2 steps:
  // 1. We blindly replace all tf ops operating on TensorList's with
  // tf_tensorlist ops. No types change in this step (so the IR is transiently
  // invalid).
  // 2. We rewrite all the types to make the IR valid again.
  //
  // The reason we need to do the rewriting this way is that not all TF variant
  // types actually represent a tensorlist. Only by looking at ops that we know
  // produce tensorlists can we deduce which TF varaints are tensorlists.
  //
  // The MLIR type conversion infrastructure doesn't handle this situation well.
  // It only knows how to handle blindly convert one type to another type.

  OwningRewritePatternList patterns;
  populateWithGenerated(&getContext(), patterns);
  patterns.insert<ConvertTfTensorlistConcatV2>(&getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<TFTensorListDialect>();
  target.addLegalDialect<TF::TensorFlowDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addIllegalOp<TF::TensorListReserveOp>();
  target.addIllegalOp<TF::TensorListGetItemOp>();
  target.addIllegalOp<TF::TensorListSetItemOp>();
  target.addIllegalOp<TF::TensorListFromTensorOp>();
  target.addIllegalOp<TF::TensorListConcatV2Op>();
  target.addIllegalOp<TF::TensorListStackOp>();

  if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    func.emitError() << "unable to lower to tf_tensorlist dialect";
    return signalPassFailure();
  }

  // The above conversions didn't do any type conversion since we don't
  // want to blindly update all variant types to tensorlist. So here we do a
  // targeted rewrite.
  auto *tfTensorListDialect =
      func.getContext()->getLoadedDialect<TFTensorListDialect>();
  auto tensorListType = TensorListType::get(func.getContext());
  SmallVector<Value, 8> typeConversionWorklist;
  func.walk([&](Operation *op) {
    if (op->getDialect() != tfTensorListDialect) {
      return;
    }
    for (auto result : op->getResults()) {
      if (isTfVariant(result.getType())) {
        result.setType(tensorListType);
        typeConversionWorklist.push_back(result);
      }
    }
  });
  while (!typeConversionWorklist.empty()) {
    Value v = typeConversionWorklist.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      // If the user is already in the tf_tensorlist dialect then skip.
      if (owner->getDialect() == tfTensorListDialect) {
        continue;
      }
      // If a user is just a terminator passing the value through a successor
      // operand, propagate through the successor operand.
      if (BranchOpInterface branchOp = dyn_cast<BranchOpInterface>(owner)) {
        if (auto arg =
                branchOp.getSuccessorBlockArgument(use.getOperandNumber())) {
          if (!arg->getType().isa<TensorListType>()) {
            arg->setType(tensorListType);
            typeConversionWorklist.push_back(*arg);
          }
          continue;
        }
      }
      // !tf.variant can have various subtypes which we blindly turn into just
      // !tf_tensorlist.list here. So elide all casts.
      if (auto castOp = dyn_cast<TF::CastOp>(owner)) {
        assert(v == castOp.x());
        castOp.y().replaceAllUsesWith(castOp.x());
        castOp.erase();
        // The RAUW could have added more uses of `v`, so put it back on the
        // worklist and process it again.
        typeConversionWorklist.push_back(v);
        break;
      }
      owner->emitError() << "unable to convert tensorlist op: "
                         << owner->getName();
      return signalPassFailure();
    }
  }
}

static PassRegistration<ConvertTFToTFTensorListPass> pass(
    "iree-tf-convert-to-tf-tensorlist",
    "Converts TF tensor list ops to the IREE tf_tensorlist dialect");

std::unique_ptr<OperationPass<FuncOp>> createConvertTFToTFTensorListPass() {
  return std::make_unique<ConvertTFToTFTensorListPass>();
}

}  // namespace tf_tensorlist
}  // namespace iree_integrations
}  // namespace mlir
