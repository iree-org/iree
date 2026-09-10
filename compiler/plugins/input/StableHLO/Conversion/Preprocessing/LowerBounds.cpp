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
void bindBounds(Value value, OpBuilder &builder) {
  auto bounds = getBounds(value.getType());
  if (!bounds) {
    return;
  }
  auto tensorType = cast<RankedTensorType>(value.getType());
  auto plainType = cast<RankedTensorType>(stripBounds(tensorType));
  value.setType(plainType);
  if (plainType.hasStaticShape()) {
    return;
  }

  Location loc = value.getLoc();
  SmallVector<Operation *> created;
  SmallVector<Value> dynamicDims;
  SmallVector<Value> dimIndices;
  for (auto [index, size] : llvm::enumerate(plainType.getShape())) {
    if (!ShapedType::isDynamic(size)) {
      continue;
    }
    auto constOp = arith::ConstantIndexOp::create(builder, loc, index);
    created.push_back(constOp);
    dimIndices.push_back(constOp);
  }
  size_t dimIndexPos = 0;
  for (auto [index, size] : llvm::enumerate(plainType.getShape())) {
    if (!ShapedType::isDynamic(size)) {
      continue;
    }
    auto dimOp =
        tensor::DimOp::create(builder, loc, value, dimIndices[dimIndexPos++]);
    created.push_back(dimOp);
    Value dim = dimOp;
    int64_t bound = bounds.getBounds()[index];
    if (!ShapedType::isDynamic(bound)) {
      auto assumption = builder.getAttr<IREE::Util::IntAssumptionAttr>(
          /*umin=*/std::nullopt, /*umax=*/static_cast<uint64_t>(bound),
          /*udiv=*/std::nullopt);
      auto assumeOp =
          IREE::Util::AssumeIntOp::create(builder, loc, dim, assumption);
      created.push_back(assumeOp);
      dim = assumeOp.getResult(0);
    }
    dynamicDims.push_back(dim);
  }
  auto tieOp = IREE::Flow::TensorTieShapeOp::create(builder, loc, plainType,
                                                    value, dynamicDims);
  created.push_back(tieOp);
  value.replaceUsesWithIf(tieOp.getResult(), [&](OpOperand &use) {
    return !llvm::is_contained(created, use.getOwner());
  });
}

struct LowerBounds final : impl::LowerBoundsBase<LowerBounds> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    // Block arguments, the function's own first.
    funcOp.walk([&](Block *block) {
      builder.setInsertionPointToStart(block);
      for (BlockArgument arg : block->getArguments()) {
        bindBounds(arg, builder);
      }
    });
    // Results, right after their producer.
    funcOp.walk([&](Operation *op) {
      builder.setInsertionPointAfter(op);
      for (Value result : op->getResults()) {
        bindBounds(result, builder);
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
