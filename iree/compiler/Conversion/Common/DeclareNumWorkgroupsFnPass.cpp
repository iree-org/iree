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
//
//===- DeclareNumWorkgroupsFnPass.cpp - Declares num_workgroups_fn --------===//
//
// Define the function that computes the number of workgroups for every entry
// point function. This pass only defines the function. Its body will be filled
// in later.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace common {

static constexpr const char kNumWorkgroupsStr[] = "__num_workgroups__";

namespace {

/// The contract between the host and the device is captured by the _impl
/// function that is called from the main entry point function. This pattern
/// looks for the call operation and
/// - Declares (doesnt define) the function that computes the number of
///   workgroups to use for this entry point function. It is defined later in
///   the codegen pipeline, when the computation is mapped to
///   workgroups/workitems. The signature of this function is
///
///      (!shapex.ranked_shape, !shapex.ranked_shape, ....) ->
///      (index, index, index)
///
///   where the arguments are the shape of the tensor inputs + outputs of the
///   dispatch region.
/// - Sets the attribute `operand_result_index` on the
///   `hal.interface.load.tensor`/`hal.interface.store.tensor` ops that are
///   later used in the generation of the function declared here.
struct DeclareNumWorkgroupsFn : OpRewritePattern<FuncOp> {
  DeclareNumWorkgroupsFn(MLIRContext *context,
                         llvm::StringRef numWorkgroupsFnAttrName,
                         PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        numWorkgroupsFnAttrName(numWorkgroupsFnAttrName) {}
  LogicalResult matchAndRewrite(FuncOp entryPointFn,
                                PatternRewriter &rewriter) const override {
    if (!isEntryPoint(entryPointFn) ||
        entryPointFn.getAttr(numWorkgroupsFnAttrName))
      return failure();
    Region &body = entryPointFn.getBody();
    if (!llvm::hasSingleElement(body)) {
      return entryPointFn.emitError(
          "unhandled dispatch function with multiple blocks");
    }
    auto callOps = body.front().getOps<CallOp>();
    if (!llvm::hasSingleElement(callOps)) {
      return entryPointFn.emitError(
          "expected dispatch function to have a single call operation");
    }
    CallOp callOp = *callOps.begin();

    SmallVector<ShapedType, 4> shapedTypes;
    shapedTypes.reserve(callOp.getNumOperands() - 1 + callOp.getNumResults());

    // Add `operand_result_index` attribute to `hal.interface.load.tensor`
    // operations that define the operands of the call op.
    for (Value operand : callOp.operands()) {
      if (!operand.getType().isa<ShapedType>()) continue;
      if (auto definingOp =
              operand.getDefiningOp<IREE::HAL::InterfaceLoadTensorOp>()) {
        definingOp.setAttr(getOperandResultNumAttrName(),
                           rewriter.getI32IntegerAttr(shapedTypes.size()));
      }
      shapedTypes.push_back(operand.getType().cast<ShapedType>());
    }

    // Add `operand_result_index` attribute to the `hal.interface.store.tensor`
    // that use the value returned by the call op.
    for (Value result : callOp.getResults()) {
      if (!result.getType().isa<ShapedType>()) continue;
      for (auto &use : result.getUses()) {
        if (auto storeOp =
                dyn_cast<IREE::HAL::InterfaceStoreTensorOp>(use.getOwner())) {
          storeOp.setAttr(getOperandResultNumAttrName(),
                          rewriter.getI32IntegerAttr(shapedTypes.size()));
        }
      }
      shapedTypes.push_back(result.getType().cast<ShapedType>());
    }

    IndexType indexType = rewriter.getIndexType();
    SmallVector<Type, 4> argTypes = llvm::to_vector<4>(
        llvm::map_range(shapedTypes, [&rewriter](ShapedType t) -> Type {
          return Shape::RankedShapeType::get(t.getShape(),
                                             rewriter.getContext());
        }));
    FuncOp numWorkgroupsFn = rewriter.create<FuncOp>(
        entryPointFn.getLoc(), entryPointFn.getName().str() + kNumWorkgroupsStr,
        rewriter.getFunctionType(argTypes, {indexType, indexType, indexType}));
    numWorkgroupsFn.setVisibility(FuncOp::Visibility::Private);
    entryPointFn.setAttr(numWorkgroupsFnAttrName,
                         rewriter.getSymbolRefAttr(numWorkgroupsFn));
    rewriter.updateRootInPlace(entryPointFn, []() {});
    return success();
  }

 private:
  llvm::StringRef numWorkgroupsFnAttrName;
};

/// Pass to define the function for number of workgroups for every entry point
/// function.
struct DeclareNumWorkgroupsFnPass
    : public PassWrapper<DeclareNumWorkgroupsFnPass, OperationPass<ModuleOp>> {
  DeclareNumWorkgroupsFnPass(llvm::StringRef numWorkgroupsFnAttrName)
      : numWorkgroupsFnAttrName(numWorkgroupsFnAttrName) {}
  DeclareNumWorkgroupsFnPass(const DeclareNumWorkgroupsFnPass &pass) {}
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
  }

  llvm::StringRef numWorkgroupsFnAttrName;
};
}  // namespace

void DeclareNumWorkgroupsFnPass::runOnOperation() {
  OwningRewritePatternList patterns;
  MLIRContext *context = &getContext();
  patterns.insert<DeclareNumWorkgroupsFn>(context, numWorkgroupsFnAttrName);
  applyPatternsAndFoldGreedily(getOperation(), patterns);
}

std::unique_ptr<OperationPass<ModuleOp>> createDeclareNumWorkgroupsFnPass(
    const llvm::StringRef numWorkgroupsFnAttrName) {
  return std::make_unique<DeclareNumWorkgroupsFnPass>(numWorkgroupsFnAttrName);
}

}  // namespace common
}  // namespace iree_compiler
}  // namespace mlir
