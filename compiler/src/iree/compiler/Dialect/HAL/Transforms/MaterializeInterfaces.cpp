// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/HAL/Analysis/BindingLayout.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hal-materialize-interfaces"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MATERIALIZEINTERFACESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// Map of original SymbolRefAttr to a list of SymbolRefAttrs in variants marked
// with the executable target the export is assigned.
using ExportExpansions = DenseMap<
    Attribute,
    SmallVector<std::pair<Attribute, IREE::HAL::ExecutableTargetAttr>>>;

// Map of operations (executables, dispatches, etc) to the executable targets
// required by those operations based on usage. If missing or empty the default
// set should be used.
using RequiredExecutableTargets =
    DenseMap<Operation *, SetVector<IREE::HAL::ExecutableTargetAttr>>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static SymbolRefAttr
makeExportSymbolRefAttr(IREE::HAL::ExecutableOp executableOp,
                        IREE::HAL::ExecutableVariantOp variantOp,
                        IREE::HAL::ExecutableExportOp exportOp) {
  return SymbolRefAttr::get(executableOp.getNameAttr(),
                            {
                                FlatSymbolRefAttr::get(variantOp.getNameAttr()),
                                FlatSymbolRefAttr::get(exportOp.getNameAttr()),
                            });
}

static void setApplicableObjects(Operation *sourceOp,
                                 IREE::HAL::ExecutableVariantOp targetOp) {
  auto objectsAttr = sourceOp->getAttrOfType<IREE::HAL::ExecutableObjectsAttr>(
      "hal.executable.objects");
  if (!objectsAttr) {
    return;
  }
  auto objects = objectsAttr.getApplicableObjects(targetOp.getTarget());
  if (!objects) {
    return;
  }
  targetOp.setObjectsAttr(*objects);
}

template <typename ExecutableOpT, typename ExportOpT>
static void
buildRequiredExecutableTypeTargetsMap(ModuleOp moduleOp,
                                      DeviceAnalysis &deviceAnalysis,
                                      BindingLayoutAnalysis &layoutAnalysis,
                                      RequiredExecutableTargets &resultMap) {
  // NOTE: we build the map before we process it so that the addresses are
  // stable.
  for (auto executableOp : moduleOp.template getOps<ExecutableOpT>()) {
    (void)resultMap[executableOp];
    for (auto exportOp : executableOp.template getOps<ExportOpT>()) {
      for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
        (void)resultMap[dispatchOp];
      }
    }
  }
  for (auto executableOp : moduleOp.template getOps<ExecutableOpT>()) {
    auto &executableTargetAttrs = resultMap[executableOp];
    for (auto exportOp : executableOp.template getOps<ExportOpT>()) {
      for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
        auto &dispatchTargetAttrs = resultMap[dispatchOp];
        deviceAnalysis.gatherRequiredExecutableTargets(dispatchOp,
                                                       dispatchTargetAttrs);
        executableTargetAttrs.insert(dispatchTargetAttrs.begin(),
                                     dispatchTargetAttrs.end());
      }
    }
    if (executableOp.isPublic()) {
      // Public executables need all possible targets.
      deviceAnalysis.gatherAllExecutableTargets(executableTargetAttrs);
    }
  }
}

// Builds a map of executable and dispatch ops to the executable targets that
// may be required.
static RequiredExecutableTargets
buildRequiredExecutableTargetsMap(ModuleOp moduleOp,
                                  DeviceAnalysis &deviceAnalysis,
                                  BindingLayoutAnalysis &layoutAnalysis) {
  RequiredExecutableTargets resultMap;
  buildRequiredExecutableTypeTargetsMap<IREE::HAL::ExecutableSourceOp,
                                        IREE::HAL::ExecutableExportOp>(
      moduleOp, deviceAnalysis, layoutAnalysis, resultMap);
  buildRequiredExecutableTypeTargetsMap<IREE::Stream::ExecutableOp,
                                        IREE::Stream::ExecutableExportOp>(
      moduleOp, deviceAnalysis, layoutAnalysis, resultMap);
  return resultMap;
}

// Updates the target entry point symbols of |dispatchOp| to the expanded set of
// variant exports in |exportExpansions|.
static void
updateDispatchTargets(IREE::Stream::CmdDispatchOp dispatchOp,
                      const ExportExpansions &exportExpansions,
                      RequiredExecutableTargets &requiredExecutableTargets) {
  auto &requiredTargetAttrs = requiredExecutableTargets[dispatchOp];
  SmallVector<Attribute> newAttrs;
  for (auto oldAttr : dispatchOp.getEntryPointRefs()) {
    auto it = exportExpansions.find(oldAttr);
    if (it == exportExpansions.end()) {
      newAttrs.push_back(oldAttr); // preserve existing
      continue;
    }
    for (auto [newAttr, targetAttr] : it->second) {
      // Filter the new expansions to only those used by the dispatch (if we
      // have a valid filter).
      if (requiredTargetAttrs.empty()) {
        newAttrs.push_back(newAttr);
      } else if (requiredTargetAttrs.contains(targetAttr)) {
        newAttrs.push_back(newAttr);
      }
    }
  }
  dispatchOp.setEntryPointsAttr(
      ArrayAttr::get(dispatchOp.getContext(), newAttrs));
}

//===----------------------------------------------------------------------===//
// hal.executable.source materialization
//===----------------------------------------------------------------------===//

static void materializeExecutableFromSourceOp(
    IREE::HAL::ExecutableSourceOp sourceOp,
    BindingLayoutAnalysis &layoutAnalysis,
    RequiredExecutableTargets &requiredExecutableTargets) {
  // Gather the required executable targets based on the dispatches to exports
  // in the source op.
  SmallVector<IREE::HAL::ExecutableTargetAttr> targetAttrs(
      requiredExecutableTargets[sourceOp].getArrayRef());
  if (targetAttrs.empty()) {
    return;
  }
  llvm::stable_sort(targetAttrs, [](auto lhs, auto rhs) {
    return lhs.getSymbolNameFragment() < rhs.getSymbolNameFragment();
  });

  // Create the op that will contain the translated executable.
  OpBuilder moduleBuilder(sourceOp);
  auto executableOp = IREE::HAL::ExecutableOp::create(
      moduleBuilder, sourceOp.getLoc(), sourceOp.getName());
  executableOp.setVisibility(sourceOp.getVisibility());

  // With this hand-authored path all variants have the same layout and entry
  // points and we can just clone them.
  auto sourceExportOps = sourceOp.getExportOps();

  // Materialize all of the hal.executable.variant ops for all backends we are
  // targeting.
  ExportExpansions exportExpansions;
  SymbolTable targetSymbolTable(executableOp);
  OpBuilder targetBuilder(&executableOp.getBlock().back());
  for (auto targetAttr : targetAttrs) {
    // Create new variant and clone the exports.
    auto targetVariantOp = IREE::HAL::ExecutableVariantOp::create(
        targetBuilder, sourceOp->getLoc(), targetAttr.getSymbolNameFragment(),
        targetAttr);
    targetSymbolTable.insert(targetVariantOp);
    OpBuilder variantBuilder(&targetVariantOp.getBlock().back());
    for (auto sourceExportOp : sourceExportOps) {
      variantBuilder.clone(*sourceExportOp);

      // Map the original export names to the new variant exports.
      auto oldRefAttr = SymbolRefAttr::get(
          executableOp.getNameAttr(),
          {FlatSymbolRefAttr::get(sourceExportOp.getNameAttr())});
      auto newRefAttr = makeExportSymbolRefAttr(executableOp, targetVariantOp,
                                                sourceExportOp);
      exportExpansions[oldRefAttr].push_back(
          std::make_pair(newRefAttr, targetAttr));
    }

    // Clone any target-specific object files specified.
    if (auto objectsAttr = sourceOp.getObjectsAttr()) {
      auto objects = objectsAttr.getApplicableObjects(targetAttr);
      if (objects) {
        targetVariantOp.setObjectsAttr(*objects);
      }
    }

    // Clone inner module contents.
    if (!sourceOp.isExternal()) {
      auto sourceModuleOp = sourceOp.getInnerModule();
      variantBuilder.clone(*sourceModuleOp);
    }
  }

  // Update all dispatch sites to reference the new expanded variants.
  for (auto exportOp : sourceExportOps) {
    for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
      updateDispatchTargets(dispatchOp, exportExpansions,
                            requiredExecutableTargets);
    }
  }

  // Remove the original.
  sourceOp.erase();
}

//===----------------------------------------------------------------------===//
// Interface definition
//===----------------------------------------------------------------------===//

// Verifies that all types used with the given entry point are supportable.
static LogicalResult
verifyEntryPointTypes(mlir::FunctionOpInterface entryFuncOp) {
  for (auto inputType : llvm::enumerate(entryFuncOp.getArgumentTypes())) {
    if (llvm::isa<IREE::Stream::BindingType>(inputType.value()) ||
        inputType.value().isInteger(32)) {
      // OK - directly translates to a HAL interface binding.
    } else {
      return entryFuncOp.emitError()
             << "unsupported interface function argument " << inputType.index()
             << " type " << inputType.value()
             << "; requires !stream.binding or i32 operands only";
    }
  }
  return success();
}

// Creates an pipeline layout attr from the analysis results.
static IREE::HAL::PipelineLayoutAttr
makePipelineLayoutAttr(const PipelineLayout &pipelineLayout,
                       IREE::HAL::ExecutableTargetAttr targetAttr,
                       OpBuilder &builder) {
  SmallVector<IREE::HAL::PipelineBindingAttr> bindingAttrs;
  for (const auto &binding : pipelineLayout.bindings) {
    bindingAttrs.push_back(IREE::HAL::PipelineBindingAttr::get(
        builder.getContext(), binding.type, binding.flags));
  }
  return IREE::HAL::PipelineLayoutAttr::get(builder.getContext(), bindingAttrs,
                                            pipelineLayout.constantCount,
                                            pipelineLayout.flags);
}

// Converts the usage of the given primitive |arg| to interface methods.
static void
convertOperandUsage(mlir::FunctionOpInterface sourceFuncOp, BlockArgument arg,
                    IREE::HAL::PipelineLayoutAttr pipelineLayoutAttr,
                    unsigned pushConstantIdx, OpBuilder &builder) {
  auto alignmentAttr = sourceFuncOp.getArgAttrOfType<IntegerAttr>(
      arg.getArgNumber(), "stream.alignment");
  auto valuesAttr = sourceFuncOp.getArgAttrOfType<ArrayAttr>(arg.getArgNumber(),
                                                             "stream.values");
  auto loadOp = IREE::HAL::InterfaceConstantLoadOp::create(
      builder, arg.getLoc(), arg.getType(), pipelineLayoutAttr,
      builder.getIndexAttr(pushConstantIdx), alignmentAttr, valuesAttr);
  arg.replaceAllUsesWith(loadOp);
}

// Converts the usage of the given !stream.binding |arg| to interface methods.
static void
convertBindingUsage(mlir::FunctionOpInterface sourceFuncOp, BlockArgument arg,
                    IREE::HAL::PipelineLayoutAttr pipelineLayoutAttr,
                    int64_t bindingOrdinal,
                    IREE::HAL::PipelineBindingAttr bindingAttr) {
  if (arg.use_empty())
    return; // no-op
  for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
    auto oldOp = dyn_cast<IREE::Stream::BindingSubspanOp>(use.getOwner());
    assert(oldOp && "bindings are only usable by stream.binding.subspan");
    OpBuilder builder(oldOp);
    auto alignmentAttr = sourceFuncOp.getArgAttrOfType<IntegerAttr>(
        arg.getArgNumber(), "stream.alignment");

    StringAttr subspanAccessAttr;

    if (auto dispatchTensorType =
            dyn_cast<IREE::TensorExt::DispatchTensorType>(oldOp.getType())) {
      if (dispatchTensorType.getAccess() ==
          IREE::TensorExt::TensorAccess::WriteOnly) {
        subspanAccessAttr = builder.getStringAttr("writeonly");
      }
    }

    auto newOp = IREE::HAL::InterfaceBindingSubspanOp::create(
        builder, oldOp.getLoc(), oldOp.getType(), pipelineLayoutAttr,
        APInt(64, bindingOrdinal), oldOp.getByteOffset(),
        oldOp.getDynamicDims(), alignmentAttr, bindingAttr.getFlags());

    if (subspanAccessAttr)
      newOp->setDiscardableAttr(kSubspanAccessAttrName, subspanAccessAttr);

    oldOp.replaceAllUsesWith(newOp.getResult());
    oldOp.erase();
  }
}

// Clones |sourceFuncOp| and updates its signature to match the |interfaceOp|
// and use the HAL interface access primitives.
static mlir::func::FuncOp
cloneFuncWithInterface(mlir::func::FuncOp sourceFuncOp,
                       const PipelineResourceMap &resourceMap,
                       IREE::HAL::PipelineLayoutAttr layoutAttr) {
  // Clone so that we can do a bunch of unsafe in-place updates.
  auto clonedFuncOp = sourceFuncOp.clone();

  // Strip all arguments as functions take all I/O through the interface API.
  clonedFuncOp.setType(FunctionType::get(clonedFuncOp.getContext(), {}, {}));

  auto *entryBlock = &clonedFuncOp.front();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Change the interface from arguments to hal.interface.* methods.
  // We do push constant compatible operands first so that they are available
  // for use by the binding accessors.
  unsigned operandIdx = 0;
  for (auto arg : entryBlock->getArguments()) {
    if (!llvm::isa<IREE::Stream::BindingType>(arg.getType())) {
      convertOperandUsage(sourceFuncOp, arg, layoutAttr, operandIdx++,
                          entryBuilder);
    }
  }
  unsigned resourceIdx = 0;
  for (auto arg : entryBlock->getArguments()) {
    if (!llvm::isa<IREE::Stream::BindingType>(arg.getType())) {
      continue; // unhandled arg type (primitive/etc)
    }
    auto binding = resourceMap[resourceIdx++];
    auto bindingAttr = layoutAttr.getBinding(binding);
    assert(bindingAttr && "layout must be consistent");
    convertBindingUsage(sourceFuncOp, arg, layoutAttr, binding, bindingAttr);
  }

  // Remove all arguments now that we've turned them into lookup ops.
  entryBlock->eraseArguments([](auto arg) { return true; });

  return clonedFuncOp;
}

// Adds the entry point ops with assigned ordinals for each entry function.
// The entry points will all use the provided |interfaceOp| and be exported with
// hal.executable.export ops.
static LogicalResult
declareEntryPointOps(IREE::Stream::ExecutableOp sourceExecutableOp,
                     IREE::HAL::ExecutableOp targetExecutableOp,
                     const BindingLayoutAnalysis &layoutAnalysis,
                     RequiredExecutableTargets &requiredExecutableTargets) {
  auto variantOps =
      targetExecutableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>();
  OpBuilder executableBuilder(&targetExecutableOp.getBlock().front());

  // Build a map of source function definitions to their version with the
  // updated interface per variant.
  DenseMap<Operation *, DenseMap<IREE::HAL::ExecutableVariantOp, Operation *>>
      targetFuncOps;
  int nextOrdinal = 0;
  for (auto exportOp : sourceExecutableOp.getBody()
                           .getOps<IREE::Stream::ExecutableExportOp>()) {
    func::FuncOp sourceFuncOp; // optional, may be extern
    if (auto sourceModuleOp = sourceExecutableOp.getInnerModule()) {
      sourceFuncOp = sourceModuleOp.lookupSymbol<mlir::func::FuncOp>(
          exportOp.getFunctionRef());
      if (failed(verifyEntryPointTypes(sourceFuncOp))) {
        return failure();
      }
    }

    // Lookup to see if a layout was specified already. If not we'll perform
    // some basic analysis to come up with our own layout.
    auto forcedLayoutAttr =
        exportOp->getAttrOfType<IREE::HAL::PipelineLayoutAttr>(
            "hal.interface.layout");
    const auto &pipelineLayout = layoutAnalysis.getPipelineLayout(exportOp);
    const auto &resourceMap = pipelineLayout.resourceMap;

    // Clone the updated function declaration into each variant.
    ExportExpansions exportExpansions;
    int ordinal = nextOrdinal++;
    for (auto variantOp : variantOps) {
      auto targetBuilder = OpBuilder::atBlockBegin(&variantOp.getBlock());

      // TODO(ravishankarm): use hal.interface.workgroup_size instead of codegen
      // attributes.
      // Check if workgroup size is set externally.
      ArrayAttr workgroupSize;
      for (auto attr : exportOp->getAttrs()) {
        if (auto exportConfig =
                dyn_cast<IREE::Codegen::ExportConfigAttr>(attr.getValue())) {
          workgroupSize = exportConfig.getWorkgroupSizeIndexArray();
          if (workgroupSize.size() < 3) {
            SmallVector<Attribute> workgroupSizeVals =
                llvm::to_vector(workgroupSize);
            workgroupSizeVals.resize(3, targetBuilder.getIndexAttr(1));
            workgroupSize = targetBuilder.getArrayAttr(workgroupSizeVals);
          }
          break;
        }
      }

      // Declare the entry point on the target.
      auto variantLayoutAttr =
          forcedLayoutAttr ? forcedLayoutAttr
                           : makePipelineLayoutAttr(pipelineLayout,
                                                    variantOp.getTargetAttr(),
                                                    targetBuilder);
      auto newExportOp = IREE::HAL::ExecutableExportOp::create(
          targetBuilder, exportOp.getLoc(),
          targetBuilder.getStringAttr(exportOp.getFunctionRef()),
          targetBuilder.getIndexAttr(ordinal), variantLayoutAttr, workgroupSize,
          /*subgroup_size=*/IntegerAttr{},
          /*workgroup_local_memory=*/IntegerAttr{});

      // Map the original export name to the new variant export.
      auto oldRefAttr =
          SymbolRefAttr::get(sourceExecutableOp.getNameAttr(),
                             {FlatSymbolRefAttr::get(exportOp.getNameAttr())});
      auto newRefAttr =
          makeExportSymbolRefAttr(targetExecutableOp, variantOp, newExportOp);
      exportExpansions[oldRefAttr].push_back(
          {newRefAttr, variantOp.getTargetAttr()});

      // Clone the workgroup count calculation function.
      if (!exportOp.getWorkgroupCount().empty()) {
        mlir::IRMapping mapper;
        exportOp.getWorkgroupCount().cloneInto(&newExportOp.getWorkgroupCount(),
                                               mapper);
        // Insert the !hal.device argument if it doesn't already exist.
        Type deviceType = targetBuilder.getType<IREE::HAL::DeviceType>();
        if (!llvm::is_contained(exportOp.getWorkgroupCount().getArgumentTypes(),
                                deviceType)) {
          newExportOp.getWorkgroupCount().insertArgument(0u, deviceType,
                                                         newExportOp.getLoc());
        }
      }

      // Clone the source function and update it to use the new interface.
      if (sourceFuncOp) {
        auto variantFuncOp = cloneFuncWithInterface(sourceFuncOp, resourceMap,
                                                    variantLayoutAttr);
        targetFuncOps[sourceFuncOp][variantOp] = variantFuncOp;
      }
    }

    // Update all dispatch sites to reference the new expanded variants.
    for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
      updateDispatchTargets(dispatchOp, exportExpansions,
                            requiredExecutableTargets);
    }
  }

  // Clone all of the ops in the source module to each variant.
  // We'll use the exported functions with the updated interfaces in place of
  // the original versions and copy everything else verbatim.
  // Note that we do this as a cleanup setup because there may be multiple
  // functions and multiple exports (with an N:M mapping) and in this way we
  // perform the variant construction in a single pass with deterministic
  // ordering that preserves the unmodified ops.
  if (auto sourceModuleOp = sourceExecutableOp.getInnerModule()) {
    for (auto variantOp : variantOps) {
      auto targetBuilder = OpBuilder::atBlockBegin(
          &variantOp.getInnerModule().getBodyRegion().front());
      for (auto &op : sourceModuleOp.getOps()) {
        auto targetVariantFuncOps = targetFuncOps.find(&op);
        if (targetVariantFuncOps != targetFuncOps.end()) {
          // Move the updated function into place.
          auto variantFuncOp = targetVariantFuncOps->second[variantOp];
          targetBuilder.insert(variantFuncOp);
        } else {
          // Regular op (globals, external function declarations, etc).
          targetBuilder.clone(op);
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.* info op conversion
//===----------------------------------------------------------------------===//

namespace {

struct ConvertReturnPattern : public OpRewritePattern<IREE::Stream::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::ReturnOp>(op, op.getOperands());
    return success();
  }
};

template <typename SrcOp, typename DstOp>
struct ConvertDispatchWorkgroupInfoPattern final
    : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getResult().getType(),
                                       op.getDimensionAttr(),
                                       /*upper_bound=*/nullptr);
    return success();
  }
};

struct InlineConstantWorkgroupSizePattern
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp sizeOp,
                                PatternRewriter &rewriter) const override {
    // Lookup the entry point matching the parent.
    auto funcOp = sizeOp->getParentOfType<mlir::func::FuncOp>();
    auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    auto exportOp = dyn_cast<IREE::HAL::ExecutableExportOp>(
        SymbolTable::lookupSymbolIn(variantOp, funcOp.getName()));
    assert(exportOp &&
           "must have an entry point corresponding to the parent func");
    auto workgroupSizeAttr = exportOp.getWorkgroupSizeAttr();
    if (!workgroupSizeAttr) {
      return failure();
    }

    uint64_t dimIdx = sizeOp.getDimension().getZExtValue();
    auto dimAttr = workgroupSizeAttr[dimIdx];
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        sizeOp, rewriter.getIndexType(), cast<TypedAttr>(dimAttr));
    return success();
  }
};

} // namespace

static LogicalResult
convertDispatchWorkgroupInfoOps(IREE::HAL::ExecutableOp executableOp) {
  RewritePatternSet patterns(executableOp.getContext());
  patterns.insert<
      ConvertReturnPattern,
      ConvertDispatchWorkgroupInfoPattern<IREE::Stream::DispatchWorkgroupIDOp,
                                          IREE::HAL::InterfaceWorkgroupIDOp>,
      ConvertDispatchWorkgroupInfoPattern<
          IREE::Stream::DispatchWorkgroupCountOp,
          IREE::HAL::InterfaceWorkgroupCountOp>,
      ConvertDispatchWorkgroupInfoPattern<IREE::Stream::DispatchWorkgroupSizeOp,
                                          IREE::HAL::InterfaceWorkgroupSizeOp>,
      InlineConstantWorkgroupSizePattern>(executableOp.getContext());
  return applyPatternsGreedily(executableOp, std::move(patterns));
}

//===----------------------------------------------------------------------===//
// --iree-hal-materialize-interfaces
//===----------------------------------------------------------------------===//

struct MaterializeInterfacesPass
    : public IREE::HAL::impl::MaterializeInterfacesPassBase<
          MaterializeInterfacesPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    BindingLayoutAnalysis layoutAnalysis(moduleOp, symbolTable);

    // Run required analysis passes.
    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // If no devices were defined and there are dispatches in the program then
    // error out. This provides a better error message than if we were to allow
    // this pass to no-op and then fail during conversion later on.
    if (layoutAnalysis.hasDispatches() &&
        deviceAnalysis.getDeviceGlobals().empty()) {
      mlir::emitError(moduleOp.getLoc())
          << "no HAL devices defined in the module; use the module-level "
             "hal.device.targets attribute, the --iree-hal-target-device= "
             "flag, or provide inputs with global !hal.devices defined";
      return signalPassFailure();
    }

    // Gather the required executable targets per executable and dispatch site.
    auto requiredExecutableTargets = buildRequiredExecutableTargetsMap(
        moduleOp, deviceAnalysis, layoutAnalysis);

    // Handle any hand-authored executables; these only need variant expansion
    // and no layout analysis as the user specified the layout themselves.
    for (auto sourceOp : llvm::make_early_inc_range(
             moduleOp.getOps<IREE::HAL::ExecutableSourceOp>())) {
      materializeExecutableFromSourceOp(sourceOp, layoutAnalysis,
                                        requiredExecutableTargets);
    }

    // Processes all executables within the input module and produce the
    // output HAL ops. We should ensure all deduping is performed prior to
    // this when it's easier to diff IR and where we still have the flow
    // context.
    for (auto sourceOp : llvm::make_early_inc_range(
             moduleOp.getOps<IREE::Stream::ExecutableOp>())) {
      auto exportOps = sourceOp.getOps<IREE::Stream::ExecutableExportOp>();
      if (exportOps.empty()) {
        continue;
      }

      // Gather a list of all #hal.executable.targets that we should produce
      // variants for based on the dispatches performed. Not all exports may be
      // used on any particular target but we let future DCE/pruning passes
      // remove them instead of modifying the inner modules here.
      SmallVector<IREE::HAL::ExecutableTargetAttr> targetAttrs(
          requiredExecutableTargets[sourceOp].getArrayRef());
      if (targetAttrs.empty()) {
        return;
      }
      llvm::stable_sort(targetAttrs, [](auto lhs, auto rhs) {
        return lhs.getSymbolNameFragment() < rhs.getSymbolNameFragment();
      });

      // Create the op that will contain the translated executable.
      OpBuilder builder = OpBuilder::atBlockEnd(moduleOp.getBody());
      builder.setInsertionPointAfter(sourceOp);
      auto executableOp = IREE::HAL::ExecutableOp::create(
          builder, sourceOp.getLoc(), sourceOp.getName());
      executableOp.setVisibility(sourceOp.getVisibility());

      // Materialize all of the hal.executable.variant ops for all backends we
      // are targeting.
      SymbolTable targetSymbolTable(executableOp);
      OpBuilder targetBuilder(&executableOp.getBlock().back());
      for (auto targetAttr : targetAttrs) {
        auto targetContainerOp = IREE::HAL::ExecutableVariantOp::create(
            targetBuilder, sourceOp->getLoc(),
            targetAttr.getSymbolNameFragment(), targetAttr);
        setApplicableObjects(sourceOp, targetContainerOp);
        targetSymbolTable.insert(targetContainerOp);
        if (sourceOp.getInnerModule()) {
          OpBuilder containerBuilder(&targetContainerOp.getBlock().back());
          mlir::ModuleOp::create(containerBuilder, sourceOp->getLoc());
        }
      }

      // Define interfaces for each exported function based on analysis.
      if (failed(declareEntryPointOps(sourceOp, executableOp, layoutAnalysis,
                                      requiredExecutableTargets))) {
        return signalPassFailure();
      }

      // Convert interface-related stream.dispatch.* ops to their
      // hal.interface.* versions.
      if (failed(convertDispatchWorkgroupInfoOps(executableOp))) {
        return signalPassFailure();
      }

      sourceOp.erase();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
