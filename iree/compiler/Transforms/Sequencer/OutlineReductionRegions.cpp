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

#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/IR/Sequencer/HLOps.h"
#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/IR/Types.h"
#include "iree/compiler/Utils/DispatchUtils.h"
#include "iree/compiler/Utils/MemRefUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Determines the shapes involved with reducing this dimension.
SmallVector<int64_t, 4> calculateResultShape(Value *input,
                                             int windowDimension) {
  SmallVector<int64_t, 4> resultShape;
  for (auto it :
       llvm::enumerate(input->getType().cast<ShapedType>().getShape())) {
    if (it.index() != windowDimension) {
      resultShape.push_back(it.value());
    }
  }
  return resultShape;
}

// Creates an executable that holds the given elemental reduction region.
// The executable will have an entry point taking the specified reduction values
// and writing the results to output arguments.
std::pair<IREE::MultiArchExecutableOp, FuncOp> createReductionExecutable(
    IREE::ReductionRegionOp regionOp, int outlinedRegionOrdinal,
    int separatedReductionIndex, int reductionDimension,
    SmallVector<Value *, 4> initialValues, SmallVector<Value *, 4> inputs) {
  Builder builder(regionOp.getContext());

  // Build function type matching 1:1 with the region signature.
  SmallVector<Type, 8> elementalOperandTypes;
  SmallVector<Type, 8> elementalResultTypes;
  for (auto *arg : regionOp.getInitialValueOperands()) {
    // (in0, in1) -> out0
    elementalOperandTypes.push_back(arg->getType());
    elementalOperandTypes.push_back(arg->getType());
    elementalResultTypes.push_back(arg->getType());
  }
  auto elementalFunctionType = FunctionType::get(
      elementalOperandTypes, elementalResultTypes, regionOp.getContext());

  // Create the executable with the region cloned into it.
  IREE::MultiArchExecutableOp multiArchExecutable;
  FuncOp elementalFunc;
  std::tie(multiArchExecutable, elementalFunc) = createRegionExecutable(
      regionOp, elementalFunctionType,
      "_reduce_" + std::to_string(outlinedRegionOrdinal) + "_dim_" +
          std::to_string(separatedReductionIndex));

  // Create a new entry point that we can use with the signature for this
  // dimension.
  SmallVector<Type, 8> allOperandTypes;
  auto inputTypes =
      llvm::map_range(inputs, [](Value *value) { return value->getType(); });
  allOperandTypes.append(inputTypes.begin(), inputTypes.end());
  auto initialValueTypes = llvm::map_range(
      initialValues, [](Value *value) { return value->getType(); });
  allOperandTypes.append(initialValueTypes.begin(), initialValueTypes.end());
  for (auto resultType : llvm::enumerate(regionOp.getResultTypes())) {
    auto shapedType = resultType.value().cast<ShapedType>();
    allOperandTypes.push_back(MemRefType::get(
        calculateResultShape(inputs[resultType.index()], reductionDimension),
        shapedType.getElementType()));
  }
  auto entryFuncType = FunctionType::get(allOperandTypes, ArrayRef<Type>{},
                                         regionOp.getContext());
  auto entryFunc =
      FuncOp::create(regionOp.getLoc(),
                     (elementalFunc.getName() + "_entry").str(), entryFuncType);
  entryFunc.setAttr("iree.executable.export",
                    UnitAttr::get(regionOp.getContext()));
  elementalFunc.getOperation()->getBlock()->push_back(entryFunc);
  entryFunc.getOperation()->moveBefore(elementalFunc);
  entryFunc.setAttr("iree.executable.reduction",
                    UnitAttr::get(regionOp.getContext()));
  entryFunc.setAttr("iree.executable.reduction.apply",
                    builder.getSymbolRefAttr(elementalFunc));

  return {multiArchExecutable, entryFunc};
}

// Converts a reduction_region into a dispatch to the outlined region function
// for a single reduction dimension.
// Returns the results of the reduction or empty if the construction fails.
SmallVector<Value *, 4> convertToDispatchOp(
    IREE::ReductionRegionOp regionOp, IREE::MultiArchExecutableOp executable,
    FuncOp entryFunc, int reductionDimension,
    SmallVector<Value *, 4> initialValues, SmallVector<Value *, 4> inputs,
    OpBuilder &dispatcherBuilder) {
  // Allocate output args and replace the return values with those.
  SmallVector<Value *, 4> resultValues;
  for (auto resultType : llvm::enumerate(regionOp.getResultTypes())) {
    // Allocate output buffer in the dispatcher to pass in to the region.
    auto shapedType = resultType.value().cast<ShapedType>();
    Value *allocatedValue = allocateDispatchOutputBuffer(
        regionOp.getLoc(),
        MemRefType::get(calculateResultShape(inputs[resultType.index()],
                                             reductionDimension),
                        shapedType.getElementType()),
        dispatcherBuilder);
    if (!allocatedValue) {
      regionOp.emitError("unable to allocate result value");
      return {};
    }
    resultValues.push_back(allocatedValue);
  }

  // Calculate workload from the result shape.
  auto *workload =
      wrapAsMemRef(calculateWorkload(regionOp, resultValues.front()), regionOp,
                   dispatcherBuilder);

  // Create the reduce op to the executable function.
  std::vector<Value *> allOperands;
  allOperands.insert(allOperands.end(), inputs.begin(), inputs.end());
  allOperands.insert(allOperands.end(), initialValues.begin(),
                     initialValues.end());
  allOperands.insert(allOperands.end(), resultValues.begin(),
                     resultValues.end());
  dispatcherBuilder.create<IREESeq::HL::DispatchOp>(
      regionOp.getLoc(), executable.getName(), entryFunc.getName(), workload,
      ArrayRef<Type>{}, allOperands);

  return resultValues;
}

// Outlines a reduction region into one or more iree.multi_arch_executables.
// This separates the reduction into multiple dispatches, one for each reduction
// dimension (thankfully XLA's operation semantics state this is ok). We then
// special case the first dispatch such that it takes the constant initial
// values so that we don't have to materialize a buffer for them.
LogicalResult outlineReductionRegion(IREE::ReductionRegionOp regionOp,
                                     int outlinedRegionOrdinal) {
  // Insert at the same place as the original region.
  OpBuilder dispatcherBuilder(regionOp);

  // Wrap input operands in memrefs.
  SmallVector<Value *, 4> initialValues{llvm::map_range(
      regionOp.getInitialValueOperands(), [&](Value *originalArg) {
        return insertDispatcherStore(regionOp, originalArg, dispatcherBuilder);
      })};
  SmallVector<Value *, 4> temps{
      llvm::map_range(regionOp.getReductionOperands(), [&](Value *originalArg) {
        return insertDispatcherStore(regionOp, originalArg, dispatcherBuilder);
      })};

  // Create one dispatch per dimension being reduced.
  // We'll do this by chaining the original input through with the temporary
  // reduction results. The results we end up with will be the originally
  // requested shape and we can just substitute them.
  if (regionOp.isWindowed()) {
    auto windowDimensions = regionOp.window_dimensions().getValue();
    auto windowStrides = regionOp.window_strides().getValue();
    auto baseDilations = regionOp.base_dilations().getValue();
    auto windowDilations = regionOp.window_dilations().getValue();
    SmallVector<std::tuple<int64_t, int64_t, int64_t, int64_t>, 4>
        sortedWindowAttrs;
    for (uint64_t i = 0; i < windowDimensions.getNumElements(); ++i) {
      int64_t windowDimension =
          windowDimensions.getValue<IntegerAttr>({i}).getInt();
      int64_t windowStride = windowStrides.getValue<IntegerAttr>({i}).getInt();
      int64_t baseDilation = baseDilations.getValue<IntegerAttr>({i}).getInt();
      int64_t windowDilation =
          windowDilations.getValue<IntegerAttr>({i}).getInt();
      sortedWindowAttrs.push_back(
          {windowDimension, windowStride, baseDilation, windowDilation});
    }
    llvm::sort(sortedWindowAttrs,
               [](std::tuple<int64_t, int64_t, int64_t, int64_t> a,
                  std::tuple<int64_t, int64_t, int64_t, int64_t> b) {
                 return std::get<0>(a) - std::get<0>(b);
               });
    for (auto windowAttrs : llvm::enumerate(sortedWindowAttrs)) {
      int64_t windowDimension = std::get<0>(windowAttrs.value());
      int64_t windowStride = std::get<1>(windowAttrs.value());
      int64_t baseDilation = std::get<2>(windowAttrs.value());
      int64_t windowDilation = std::get<3>(windowAttrs.value());
      IREE::MultiArchExecutableOp multiArchExecutable;
      FuncOp entryFunc;
      std::tie(multiArchExecutable, entryFunc) = createReductionExecutable(
          regionOp, outlinedRegionOrdinal, windowAttrs.index(), windowDimension,
          initialValues, temps);
      entryFunc.setAttr("iree.executable.reduction.padding_mode",
                        dispatcherBuilder.getI32IntegerAttr(
                            regionOp.padding_mode().getValue()));
      entryFunc.setAttr("iree.executable.reduction.window_dimension",
                        dispatcherBuilder.getI32IntegerAttr(windowDimension));
      entryFunc.setAttr("iree.executable.reduction.window_stride",
                        dispatcherBuilder.getI32IntegerAttr(windowStride));
      entryFunc.setAttr("iree.executable.reduction.base_dilation",
                        dispatcherBuilder.getI32IntegerAttr(baseDilation));
      entryFunc.setAttr("iree.executable.reduction.window_dilation",
                        dispatcherBuilder.getI32IntegerAttr(windowDilation));
      temps = convertToDispatchOp(regionOp, multiArchExecutable, entryFunc,
                                  windowDimension, initialValues,
                                  std::move(temps), dispatcherBuilder);
      if (temps.empty()) {
        return regionOp.emitOpError()
               << "Failed to construct reduction for windowed dimension "
               << windowDimension;
      }
    }
  } else {
    auto dimensions = regionOp.dimensions().getValue();
    SmallVector<int64_t, 4> sortedDimensions;
    for (uint64_t i = 0; i < dimensions.getNumElements(); ++i) {
      sortedDimensions.push_back(
          dimensions.getValue<IntegerAttr>({i}).getInt());
    }
    llvm::sort(sortedDimensions, [](int64_t a, int64_t b) { return a - b; });
    for (auto dimension : llvm::enumerate(sortedDimensions)) {
      IREE::MultiArchExecutableOp multiArchExecutable;
      FuncOp entryFunc;
      std::tie(multiArchExecutable, entryFunc) = createReductionExecutable(
          regionOp, outlinedRegionOrdinal, dimension.index(), dimension.value(),
          initialValues, temps);
      entryFunc.setAttr("iree.executable.reduction.dimension",
                        dispatcherBuilder.getI32IntegerAttr(dimension.value()));
      temps = convertToDispatchOp(regionOp, multiArchExecutable, entryFunc,
                                  dimension.value(), initialValues,
                                  std::move(temps), dispatcherBuilder);
      if (temps.empty()) {
        return regionOp.emitOpError()
               << "Failed to construct reduction for dimension "
               << dimension.value();
      }
    }
  }
  for (auto it : llvm::enumerate(regionOp.getResults())) {
    insertDispatcherLoad(regionOp, it.value(), temps[it.index()],
                         dispatcherBuilder);
  }

  // Erase original region.
  regionOp.erase();

  return success();
}

}  // namespace

class OutlineReductionRegionsPass
    : public ModulePass<OutlineReductionRegionsPass> {
 public:
  void runOnModule() override {
    auto module = getModule();

    SymbolTable symbolTable(module);
    auto funcs = module.getOps<FuncOp>();
    SmallVector<FuncOp, 4> funcOps(funcs.begin(), funcs.end());
    for (auto func : funcOps) {
      // Outline all of the iree.reduction_region ops in this function.
      std::vector<IREE::ReductionRegionOp> reductionRegionOps;
      func.walk([&](IREE::ReductionRegionOp op) {
        reductionRegionOps.push_back(op);
      });
      for (int i = 0; i < reductionRegionOps.size(); ++i) {
        if (failed(outlineReductionRegion(reductionRegionOps[i], i))) {
          return signalPassFailure();
        }
      }
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createOutlineReductionRegionsPass() {
  return std::make_unique<OutlineReductionRegionsPass>();  // NOLINT
}

static PassRegistration<OutlineReductionRegionsPass> pass(
    "iree-outline-reduction-regions",
    "Outlines reduction regions into standalone functions");

}  // namespace iree_compiler
}  // namespace mlir
