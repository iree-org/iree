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

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Utils/DispatchUtils.h"
#include "iree/compiler/Dialect/Flow/Utils/WorkloadUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Determines the shapes involved with reducing this dimension.
SmallVector<int64_t, 4> calculateResultShape(Value input, int windowDimension) {
  SmallVector<int64_t, 4> resultShape;
  for (auto it :
       llvm::enumerate(input.getType().cast<ShapedType>().getShape())) {
    if (it.index() != windowDimension) {
      resultShape.push_back(it.value());
    }
  }
  return resultShape;
}

// Converts a reduction_region into a dispatch to the outlined region function
// for a single reduction dimension.
// Returns the results of the reduction or empty if the construction fails.
SmallVector<Value, 4> convertToDispatchOp(
    Operation *regionOp, ExecutableOp executableOp, StringRef entryPointName,
    int reductionDimension, SmallVector<Value, 4> initialValues,
    SmallVector<Value, 4> inputs, OpBuilder &dispatcherBuilder) {
  SmallVector<Type, 4> resultTypes;
  for (auto resultType : llvm::enumerate(regionOp->getResultTypes())) {
    // Allocate output buffer in the dispatcher to pass in to the region.
    auto shapedType = resultType.value().cast<ShapedType>();
    auto reducedType = RankedTensorType::get(
        calculateResultShape(inputs[resultType.index()], reductionDimension),
        shapedType.getElementType());
    resultTypes.push_back(reducedType);
  }

  // Calculate workload from the result shape.
  auto workload =
      calculateWorkload(regionOp, resultTypes.front().cast<ShapedType>());

  // Create the reduce op to the executable function.
  std::vector<Value> allOperands;
  allOperands.insert(allOperands.end(), inputs.begin(), inputs.end());
  allOperands.insert(allOperands.end(), initialValues.begin(),
                     initialValues.end());
  auto dispatchOp = dispatcherBuilder.create<DispatchOp>(
      regionOp->getLoc(), executableOp.getName(), entryPointName, workload,
      resultTypes, allOperands);

  return llvm::to_vector<4>(dispatchOp.getResults());
}

// Creates an executable that holds the given elemental reduction region.
// The executable will have an entry point taking the specified reduction values
// and writing the results to output arguments.
std::pair<ExecutableOp, ReductionEntryOp> createReductionExecutable(
    ReductionRegionOp regionOp, int outlinedRegionOrdinal,
    int separatedReductionIndex, int reductionDimension,
    SmallVector<Value, 4> initialValues, SmallVector<Value, 4> inputs,
    llvm::StringMap<FuncOp> &dispatchableFuncOps) {
  // Build function type matching 1:1 with the region signature.
  SmallVector<Type, 8> elementalOperandTypes;
  SmallVector<Type, 8> elementalResultTypes;
  for (auto arg : regionOp.initial_values()) {
    // (in0, in1) -> out0
    elementalOperandTypes.push_back(arg.getType());
    elementalOperandTypes.push_back(arg.getType());
    elementalResultTypes.push_back(arg.getType());
  }
  auto elementalFunctionType = FunctionType::get(
      elementalOperandTypes, elementalResultTypes, regionOp.getContext());

  // Create the executable with the region cloned into it.
  ExecutableOp executableOp;
  FuncOp elementalFuncOp;
  std::tie(executableOp, elementalFuncOp) = createRegionExecutable(
      regionOp, elementalFunctionType,
      "_reduce_" + std::to_string(outlinedRegionOrdinal) + "_dim_" +
          std::to_string(separatedReductionIndex),
      dispatchableFuncOps);

  // Create a new entry point that we can use with the signature for this
  // dimension.
  SmallVector<Type, 8> allOperandTypes;
  auto inputTypes =
      llvm::map_range(inputs, [](Value value) { return value.getType(); });
  allOperandTypes.append(inputTypes.begin(), inputTypes.end());
  auto initialValueTypes = llvm::map_range(
      initialValues, [](Value value) { return value.getType(); });
  allOperandTypes.append(initialValueTypes.begin(), initialValueTypes.end());
  SmallVector<Type, 4> resultTypes;
  for (auto resultType : llvm::enumerate(regionOp.getResultTypes())) {
    auto shapedType = resultType.value().cast<ShapedType>();
    auto reducedType = RankedTensorType::get(
        calculateResultShape(inputs[resultType.index()], reductionDimension),
        shapedType.getElementType());
    resultTypes.push_back(reducedType);
  }
  auto entryFuncType =
      FunctionType::get(allOperandTypes, resultTypes, regionOp.getContext());
  auto entryFuncOp = FuncOp::create(
      regionOp.getLoc(), (elementalFuncOp.getName() + "_entry").str(),
      entryFuncType);
  elementalFuncOp.getOperation()->getBlock()->push_back(entryFuncOp);
  entryFuncOp.getOperation()->moveBefore(elementalFuncOp);

  // Add dispatch export pointing at the function.
  OpBuilder builder(executableOp.body());
  auto entryPointOp = builder.create<ReductionEntryOp>(
      regionOp.getLoc(), builder.getStringAttr(entryFuncOp.getName()),
      builder.getSymbolRefAttr(entryFuncOp),
      builder.getSymbolRefAttr(elementalFuncOp),
      builder.getI32IntegerAttr(reductionDimension));

  return {executableOp, entryPointOp};
}

// Outlines a reduction region into one or more executables.
// This separates the reduction into multiple dispatches, one for each reduction
// dimension (thankfully XLA's operation semantics state this is ok). We then
// special case the first dispatch such that it takes the constant initial
// values so that we don't have to materialize a buffer for them.
LogicalResult outlineReductionRegion(
    ReductionRegionOp regionOp, int outlinedRegionOrdinal,
    llvm::StringMap<FuncOp> &dispatchableFuncOps) {
  // Insert at the same place as the original region.
  OpBuilder dispatcherBuilder(regionOp);

  SmallVector<Value, 4> temps{regionOp.operands()};

  // Create one dispatch per dimension being reduced.
  // We'll do this by chaining the original input through with the temporary
  // reduction results. The results we end up with will be the originally
  // requested shape and we can just substitute them.
  auto dimensions = regionOp.dimensions().getValue();
  SmallVector<int32_t, 4> sortedDimensions;
  for (uint32_t i = 0; i < dimensions.getNumElements(); ++i) {
    sortedDimensions.push_back(dimensions.getValue<IntegerAttr>({i}).getInt());
  }
  llvm::sort(sortedDimensions, [](int32_t a, int32_t b) { return a - b; });
  for (auto dimension : llvm::enumerate(sortedDimensions)) {
    // Create the executable with the region cloned into it.
    ExecutableOp executableOp;
    ReductionEntryOp entryPointOp;
    std::tie(executableOp, entryPointOp) = createReductionExecutable(
        regionOp, outlinedRegionOrdinal, dimension.index(), dimension.value(),
        regionOp.initial_values(), temps, dispatchableFuncOps);

    // Finally convert the dispatch region into a dispatch to the outlined func.
    temps = convertToDispatchOp(regionOp, executableOp, entryPointOp.getName(),
                                dimension.value(), regionOp.initial_values(),
                                std::move(temps), dispatcherBuilder);
    if (temps.empty()) {
      return regionOp.emitOpError()
             << "failed to construct reduction for dimension "
             << dimension.value();
    }
  }

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i).replaceAllUsesWith(temps[i]);
  }

  // Erase original region.
  regionOp.erase();

  return success();
}

// Creates an executable that holds the given elemental reduction region.
// The executable will have an entry point taking the specified reduction values
// and writing the results to output arguments.
std::pair<ExecutableOp, WindowedReductionEntryOp>
createWindowedReductionExecutable(
    WindowedReductionRegionOp regionOp, int outlinedRegionOrdinal,
    int separatedReductionIndex, int32_t windowDimension, int32_t windowStride,
    int32_t baseDilation, int32_t windowDilation,
    SmallVector<Value, 4> initialValues, SmallVector<Value, 4> inputs,
    llvm::StringMap<FuncOp> &dispatchableFuncOps) {
  // Build function type matching 1:1 with the region signature.
  SmallVector<Type, 8> elementalOperandTypes;
  SmallVector<Type, 8> elementalResultTypes;
  for (auto arg : regionOp.initial_values()) {
    // (in0, in1) -> out0
    elementalOperandTypes.push_back(arg.getType());
    elementalOperandTypes.push_back(arg.getType());
    elementalResultTypes.push_back(arg.getType());
  }
  auto elementalFunctionType = FunctionType::get(
      elementalOperandTypes, elementalResultTypes, regionOp.getContext());

  // Create the executable with the region cloned into it.
  ExecutableOp executableOp;
  FuncOp elementalFuncOp;
  std::tie(executableOp, elementalFuncOp) = createRegionExecutable(
      regionOp, elementalFunctionType,
      "_reduce_" + std::to_string(outlinedRegionOrdinal) + "_dim_" +
          std::to_string(separatedReductionIndex),
      dispatchableFuncOps);

  // Create a new entry point that we can use with the signature for this
  // dimension.
  SmallVector<Type, 8> allOperandTypes;
  auto inputTypes =
      llvm::map_range(inputs, [](Value value) { return value.getType(); });
  allOperandTypes.append(inputTypes.begin(), inputTypes.end());
  auto initialValueTypes = llvm::map_range(
      initialValues, [](Value value) { return value.getType(); });
  allOperandTypes.append(initialValueTypes.begin(), initialValueTypes.end());
  SmallVector<Type, 4> resultTypes;
  for (auto resultType : llvm::enumerate(regionOp.getResultTypes())) {
    auto shapedType = resultType.value().cast<ShapedType>();
    auto reducedType = RankedTensorType::get(
        calculateResultShape(inputs[resultType.index()], windowDimension),
        shapedType.getElementType());
    resultTypes.push_back(reducedType);
  }
  auto entryFuncType =
      FunctionType::get(allOperandTypes, resultTypes, regionOp.getContext());
  auto entryFuncOp = FuncOp::create(
      regionOp.getLoc(), (elementalFuncOp.getName() + "_entry").str(),
      entryFuncType);
  elementalFuncOp.getOperation()->getBlock()->push_back(entryFuncOp);
  entryFuncOp.getOperation()->moveBefore(elementalFuncOp);

  // Add dispatch export pointing at the function.
  OpBuilder builder(executableOp.body());
  auto entryPointOp = builder.create<WindowedReductionEntryOp>(
      regionOp.getLoc(), builder.getStringAttr(entryFuncOp.getName()),
      builder.getSymbolRefAttr(entryFuncOp),
      builder.getSymbolRefAttr(elementalFuncOp),
      builder.getI32IntegerAttr(windowDimension),
      builder.getI32IntegerAttr(windowStride),
      builder.getI32IntegerAttr(baseDilation),
      builder.getI32IntegerAttr(windowDilation),
      builder.getI32IntegerAttr(
          static_cast<uint32_t>(regionOp.padding_mode())));

  return {executableOp, entryPointOp};
}

// Outlines a windowed reduction region into one or more executables.
// This separates the reduction into multiple dispatches, one for each reduction
// dimension (thankfully XLA's operation semantics state this is ok). We then
// special case the first dispatch such that it takes the constant initial
// values so that we don't have to materialize a buffer for them.
LogicalResult outlineWindowedReductionRegion(
    WindowedReductionRegionOp regionOp, int outlinedRegionOrdinal,
    llvm::StringMap<FuncOp> &dispatchableFuncOps) {
  // Insert at the same place as the original region.
  OpBuilder dispatcherBuilder(regionOp);

  SmallVector<Value, 4> initialValues{regionOp.initial_values()};
  SmallVector<Value, 4> temps{regionOp.operands()};

  // Create one dispatch per dimension being reduced.
  // We'll do this by chaining the original input through with the temporary
  // reduction results. The results we end up with will be the originally
  // requested shape and we can just substitute them.
  using WindowTuple = std::tuple<int32_t, int32_t, int32_t, int32_t>;

  auto windowDimensions = regionOp.window_dimensions();
  auto windowStrides = regionOp.window_strides();
  auto baseDilations = regionOp.base_dilations();
  auto windowDilations = regionOp.window_dilations();
  SmallVector<WindowTuple, 4> sortedWindowAttrs;
  for (uint32_t i = 0; i < windowDimensions.getNumElements(); ++i) {
    int32_t windowDimension =
        windowDimensions.getValue<IntegerAttr>({i}).getInt();
    int32_t windowStride = windowStrides.getValue<IntegerAttr>({i}).getInt();
    int32_t baseDilation = baseDilations.getValue<IntegerAttr>({i}).getInt();
    int32_t windowDilation =
        windowDilations.getValue<IntegerAttr>({i}).getInt();
    sortedWindowAttrs.push_back(WindowTuple(windowDimension, windowStride,
                                            baseDilation, windowDilation));
  }
  llvm::sort(sortedWindowAttrs, [](WindowTuple a, WindowTuple b) {
    return std::get<0>(a) - std::get<0>(b);
  });
  for (auto windowAttrs : llvm::enumerate(sortedWindowAttrs)) {
    int32_t windowDimension = std::get<0>(windowAttrs.value());
    int32_t windowStride = std::get<1>(windowAttrs.value());
    int32_t baseDilation = std::get<2>(windowAttrs.value());
    int32_t windowDilation = std::get<3>(windowAttrs.value());
    ExecutableOp executableOp;
    WindowedReductionEntryOp entryPointOp;
    std::tie(executableOp, entryPointOp) = createWindowedReductionExecutable(
        regionOp, outlinedRegionOrdinal, windowAttrs.index(), windowDimension,
        windowStride, baseDilation, windowDilation, initialValues, temps,
        dispatchableFuncOps);
    temps = convertToDispatchOp(regionOp, executableOp, entryPointOp.getName(),
                                windowDimension, initialValues,
                                std::move(temps), dispatcherBuilder);
    if (temps.empty()) {
      return regionOp.emitOpError()
             << "failed to construct reduction for windowed dimension "
             << windowDimension;
    }
  }

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i).replaceAllUsesWith(temps[i]);
  }

  // Erase original region.
  regionOp.erase();

  return success();
}

}  // namespace

class OutlineReductionRegionsPass
    : public ModulePass<OutlineReductionRegionsPass> {
 public:
  OutlineReductionRegionsPass() = default;
  explicit OutlineReductionRegionsPass(
      std::shared_ptr<llvm::StringMap<FuncOp>> dispatchableFuncOps)
      : dispatchableFuncOps_(std::move(dispatchableFuncOps)) {}

  void runOnModule() override {
    // TODO(benvanik): replace with a pattern rewriter?
    auto funcOps = llvm::to_vector<32>(getModule().getOps<FuncOp>());
    for (auto funcOp : funcOps) {
      SmallVector<ReductionRegionOp, 4> reductionRegionOps;
      funcOp.walk(
          [&](ReductionRegionOp op) { reductionRegionOps.push_back(op); });
      for (int i = 0; i < reductionRegionOps.size(); ++i) {
        if (failed(outlineReductionRegion(reductionRegionOps[i], i,
                                          *dispatchableFuncOps_))) {
          return signalPassFailure();
        }
      }
      SmallVector<WindowedReductionRegionOp, 4> windowedReductionRegionOps;
      funcOp.walk([&](WindowedReductionRegionOp op) {
        windowedReductionRegionOps.push_back(op);
      });
      for (int i = 0; i < windowedReductionRegionOps.size(); ++i) {
        if (failed(outlineWindowedReductionRegion(windowedReductionRegionOps[i],
                                                  i, *dispatchableFuncOps_))) {
          return signalPassFailure();
        }
      }
    }
  }

 private:
  std::shared_ptr<llvm::StringMap<FuncOp>> dispatchableFuncOps_;
};

std::unique_ptr<OpPassBase<ModuleOp>> createOutlineReductionRegionsPass(
    std::shared_ptr<llvm::StringMap<FuncOp>> dispatchableFuncOps) {
  return std::make_unique<OutlineReductionRegionsPass>(
      std::move(dispatchableFuncOps));  // NOLINT
}

static PassRegistration<OutlineReductionRegionsPass> pass(
    "iree-flow-outline-reduction-regions",
    "Outlines reduction regions into standalone functions");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
