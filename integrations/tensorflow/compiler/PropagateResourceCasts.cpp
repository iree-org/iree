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

#include "integrations/tensorflow/compiler/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace iree_compiler {

namespace {
// Determine whether we should bypass the cast for input (a) to output (b).
bool shouldBypassCast(ShapedType a, ShapedType b) {
  // If the element type changes the cast is required.
  if (a.getElementType() != b.getElementType()) {
    return false;
  }

  // If we have no rank for the output we should bypass the cast.
  if (!b.hasRank()) {
    return true;
  }

  // If the input doesn't have a rank we can't gain informatio
  if (!a.hasRank()) {
    return false;
  }

  if (a.getRank() != b.getRank()) {
    return false;
  }

  auto a_shape = a.getShape();
  auto b_shape = b.getShape();
  for (auto pair : llvm::zip(a_shape, b_shape)) {
    auto a_dim = std::get<0>(pair);
    auto b_dim = std::get<1>(pair);
    if (a_dim != b_dim && a_dim == -1) {
      return false;
    }
  }
  return true;
}
}  // namespace

// Attempts to propagate resource casts by bypassing them when they are not
// necessary or can further refine required types.
class PropagateResourceCasts
    : public PassWrapper<PropagateResourceCasts, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    auto operation = getOperation();
    for (auto func : operation.getOps<FuncOp>()) {
      for (auto cast : func.getOps<TF::CastOp>()) {
        auto input = cast.x();
        auto output = cast.getResult();

        auto input_ty = input.getType().cast<ShapedType>();
        auto output_ty = output.getType().cast<ShapedType>();

        // If the input/output types match we can just bypass it.
        if (input_ty == output_ty) {
          output.replaceAllUsesWith(input);
          continue;
        }

        auto input_element_ty =
            input_ty.getElementType().dyn_cast<TF::ResourceType>();
        auto output_element_ty =
            output_ty.getElementType().dyn_cast<TF::ResourceType>();

        // Check whether it is a
        if (!input_element_ty || !output_element_ty ||
            input_element_ty.getSubtypes().empty()) {
          continue;
        }

        auto input_resource_ty = input_element_ty.getSubtypes().front();
        if (!output_element_ty.getSubtypes().empty()) {
          auto output_resource_ty = output_element_ty.getSubtypes().front();
          if (!shouldBypassCast(input_resource_ty, output_resource_ty)) {
            continue;
          }
        }

        // TODO(suderman): Check which functions could be updated and
        // substitute.
        output.replaceAllUsesWith(input);
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPropagateResourceCasts() {
  return std::make_unique<PropagateResourceCasts>();
}

static PassRegistration<PropagateResourceCasts> pass(
    "iree-propagate-resource-casts",
    "Guarantee all func's have only a single use.");

}  // namespace iree_compiler
}  // namespace mlir
