// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowers `#stablehlo.bounds` to a `util.assume.int` on each bounded
// dimension, tied back to the tensor.

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LOWERBOUNDS
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

namespace {

mlir::stablehlo::TypeExtensionsAttr getBounds(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return nullptr;
  }
  return dyn_cast_if_present<mlir::stablehlo::TypeExtensionsAttr>(
      tensorType.getEncoding());
}

Type stripBounds(Type type) {
  if (!getBounds(type)) {
    return type;
  }
  auto tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType());
}

// Rewrites `value` to its plain type and routes its uses through a
// `flow.tensor.tie_shape` whose dynamic dims carry the bounds.
void applyBounds(OpBuilder &builder, Value value) {
  auto bounds = getBounds(value.getType());
  if (!bounds) {
    return;
  }
  auto tensorType = cast<RankedTensorType>(value.getType());
  auto plainType =
      RankedTensorType::get(tensorType.getShape(), tensorType.getElementType());
  value.setType(plainType);
  if (plainType.hasStaticShape()) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(value);

  SmallVector<OpOperand *> originalUses;
  for (OpOperand &use : value.getUses()) {
    originalUses.push_back(&use);
  }

  Location loc = value.getLoc();
  SmallVector<Value> dynamicDims;
  for (auto [index, size] : llvm::enumerate(plainType.getShape())) {
    if (!ShapedType::isDynamic(size)) {
      continue;
    }
    auto constOp = arith::ConstantIndexOp::create(builder, loc, index);
    auto dimOp = tensor::DimOp::create(builder, loc, value, constOp);
    Value dim = dimOp;
    int64_t bound = bounds.getBounds()[index];
    if (!ShapedType::isDynamic(bound)) {
      auto assumption = builder.getAttr<IREE::Util::IntAssumptionAttr>(
          /*umin=*/std::nullopt, /*umax=*/static_cast<uint64_t>(bound),
          /*udiv=*/std::nullopt);
      auto assumeOp =
          IREE::Util::AssumeIntOp::create(builder, loc, dim, assumption);
      dim = assumeOp.getResult(0);
    }
    dynamicDims.push_back(dim);
  }
  auto tieOp = IREE::Flow::TensorTieShapeOp::create(builder, loc, plainType,
                                                    value, dynamicDims);
  for (OpOperand *use : originalUses) {
    use->set(tieOp.getResult());
  }
}

struct LowerBounds final : impl::LowerBoundsBase<LowerBounds> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());
    AttrTypeReplacer constantAttrReplacer;
    constantAttrReplacer.addReplacement(
        [](DenseElementsAttr value) -> Attribute {
          auto plainType = cast<ShapedType>(stripBounds(value.getType()));
          if (plainType == value.getType()) {
            return value;
          }
          return value.reshape(plainType);
        });

    funcOp.walk([&](Block *block) {
      for (BlockArgument arg : block->getArguments()) {
        applyBounds(builder, arg);
      }
    });
    funcOp.walk([&](Operation *op) {
      // Constants infer their result types from their value attributes. Keep
      // those types consistent when removing the encoding from the results.
      if (op->hasTrait<OpTrait::ConstantLike>()) {
        constantAttrReplacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                               /*replaceLocs=*/false,
                                               /*replaceTypes=*/false);
      }
      for (Value result : op->getResults()) {
        applyBounds(builder, result);
      }
    });

    // The signature is not reached by the walks above.
    FunctionType oldType = funcOp.getFunctionType();
    funcOp.setType(FunctionType::get(
        funcOp.getContext(),
        llvm::map_to_vector(oldType.getInputs(), stripBounds),
        llvm::map_to_vector(oldType.getResults(), stripBounds)));
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
