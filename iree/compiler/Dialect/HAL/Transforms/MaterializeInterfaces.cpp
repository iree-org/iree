// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/Analysis/BindingLayout.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hal-materialize-interfaces"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace {

//===----------------------------------------------------------------------===//
// hal.executable.source materialization
//===----------------------------------------------------------------------===//

static LogicalResult materializeExecutableFromSourceOp(
    IREE::HAL::ExecutableSourceOp sourceOp,
    ArrayRef<IREE::HAL::ExecutableTargetAttr> targetAttrs) {
  OpBuilder moduleBuilder(sourceOp);

  // Create the op that will contain the translated executable.
  auto executableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
      sourceOp.getLoc(), sourceOp.getName());
  executableOp.setVisibility(sourceOp.getVisibility());

  // With this hand-authored path all variants have the same layout and entry
  // points and we can just clone them.
  auto sourceEntryPointOps =
      sourceOp.getOps<IREE::HAL::ExecutableEntryPointOp>();
  auto sourceModuleOp = sourceOp.getInnerModule();

  // Materialize all of the hal.executable.variant ops for all backends we are
  // targeting.
  SymbolTable targetSymbolTable(executableOp);
  OpBuilder targetBuilder(&executableOp.getBlock().back());
  for (auto targetAttr : targetAttrs) {
    auto targetVariantOp = targetBuilder.create<IREE::HAL::ExecutableVariantOp>(
        sourceOp->getLoc(), targetAttr.getSymbolNameFragment(), targetAttr);
    targetSymbolTable.insert(targetVariantOp);
    OpBuilder variantBuilder(&targetVariantOp.getBlock().back());
    for (auto sourceEntryPointOp : sourceEntryPointOps) {
      variantBuilder.clone(*sourceEntryPointOp);
    }
    variantBuilder.clone(*sourceModuleOp);
  }
  // Remove the original.
  sourceOp.erase();

  return success();
}

static LogicalResult materializeExecutablesFromSourceOps(
    mlir::ModuleOp moduleOp) {
  auto sourceOps =
      llvm::to_vector<32>(moduleOp.getOps<IREE::HAL::ExecutableSourceOp>());
  for (auto sourceOp : sourceOps) {
    // Gather a list of all #hal.executable.targets that we should produce
    // variants for.
    auto targetAttrs =
        IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(sourceOp);
    if (targetAttrs.empty()) {
      return sourceOp.emitError()
             << "no executable targets specified for translation";
    }

    if (failed(materializeExecutableFromSourceOp(sourceOp, targetAttrs))) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Interface definition
//===----------------------------------------------------------------------===//

// Verifies that all types used with the given entry point are supportable.
static LogicalResult verifyEntryPointTypes(mlir::FuncOp entryFuncOp) {
  for (auto inputType : llvm::enumerate(entryFuncOp.getType().getInputs())) {
    if (inputType.value().isa<IREE::Stream::BindingType>() ||
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

// Creates an executable layout attr from the analysis results.
static IREE::HAL::ExecutableLayoutAttr makeExecutableLayoutAttr(
    const ExecutableLayout &executableLayout, OpBuilder &builder) {
  SmallVector<IREE::HAL::DescriptorSetLayoutAttr> setLayoutAttrs;
  for (const auto &setLayout : executableLayout.setLayouts) {
    SmallVector<IREE::HAL::DescriptorSetBindingAttr> bindingAttrs;
    for (const auto &binding : setLayout.bindings) {
      bindingAttrs.push_back(IREE::HAL::DescriptorSetBindingAttr::get(
          builder.getContext(), binding.ordinal, binding.type));
    }
    setLayoutAttrs.push_back(IREE::HAL::DescriptorSetLayoutAttr::get(
        builder.getContext(), setLayout.ordinal, bindingAttrs));
  }
  return IREE::HAL::ExecutableLayoutAttr::get(
      builder.getContext(), executableLayout.pushConstantCount, setLayoutAttrs);
}

// Converts the usage of the given primitive |arg| to interface methods.
static void convertOperandUsage(mlir::FuncOp sourceFuncOp, BlockArgument arg,
                                unsigned pushConstantIdx, OpBuilder &builder) {
  auto alignmentAttr = sourceFuncOp.getArgAttrOfType<IntegerAttr>(
      arg.getArgNumber(), "stream.alignment");
  auto valuesAttr = sourceFuncOp.getArgAttrOfType<ArrayAttr>(arg.getArgNumber(),
                                                             "stream.values");
  auto loadOp = builder.create<IREE::HAL::InterfaceConstantLoadOp>(
      arg.getLoc(), arg.getType(), builder.getIndexAttr(pushConstantIdx),
      alignmentAttr, valuesAttr);
  arg.replaceAllUsesWith(loadOp);
}

// Converts the usage of the given !stream.binding |arg| to interface methods.
static void convertBindingUsage(
    mlir::FuncOp sourceFuncOp, BlockArgument arg,
    IREE::HAL::DescriptorSetLayoutAttr setLayoutAttr,
    IREE::HAL::DescriptorSetBindingAttr bindingAttr) {
  if (arg.use_empty()) return;  // no-op
  for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
    auto oldOp = dyn_cast<IREE::Stream::BindingSubspanOp>(use.getOwner());
    assert(oldOp && "bindings are only usable by stream.binding.subspan");
    OpBuilder builder(oldOp);
    auto alignmentAttr = sourceFuncOp.getArgAttrOfType<IntegerAttr>(
        arg.getArgNumber(), "stream.alignment");
    auto newOp = builder.create<IREE::HAL::InterfaceBindingSubspanOp>(
        oldOp.getLoc(), oldOp.getType(), APInt(64, setLayoutAttr.getOrdinal()),
        APInt(64, bindingAttr.getOrdinal()), bindingAttr.getType(),
        oldOp.byte_offset(), oldOp.dynamic_dims(), alignmentAttr);
    oldOp.replaceAllUsesWith(newOp.result());
    oldOp.erase();
  }
}

// Clones |sourceFuncOp| and updates its signature to match the |interfaceOp|
// and use the HAL interface access primitives.
static mlir::FuncOp cloneFuncWithInterface(
    mlir::FuncOp sourceFuncOp, const ExecutableLayout &executableLayout,
    IREE::HAL::ExecutableLayoutAttr layoutAttr) {
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
    if (!arg.getType().isa<IREE::Stream::BindingType>()) {
      convertOperandUsage(sourceFuncOp, arg, operandIdx++, entryBuilder);
    }
  }
  unsigned resourceIdx = 0;
  for (auto arg : entryBlock->getArguments()) {
    if (!arg.getType().isa<IREE::Stream::BindingType>()) continue;
    auto setBinding = executableLayout.resourceMap[resourceIdx++];
    auto setLayoutAttr = layoutAttr.getSetLayouts()[setBinding.first];
    auto bindingAttr = setLayoutAttr.getBindings()[setBinding.second];
    convertBindingUsage(sourceFuncOp, arg, setLayoutAttr, bindingAttr);
  }

  // Remove all arguments now that we've turned them into lookup ops.
  entryBlock->eraseArguments([](auto arg) { return true; });

  return clonedFuncOp;
}

// Annotates |dispatchOp| with resource binding to interface binding mappings.
// TODO(benvanik): have a HAL op with structured information instead.
static void annotateDispatchSite(IREE::Stream::CmdDispatchOp dispatchOp,
                                 const ExecutableLayout &executableLayout) {
  SmallVector<Attribute> bindingAttrs;
  for (auto setBinding : executableLayout.resourceMap) {
    bindingAttrs.push_back(IREE::HAL::InterfaceBindingAttr::get(
        dispatchOp.getContext(), setBinding.first, setBinding.second));
  }
  dispatchOp->setAttr("hal.interface.bindings",
                      ArrayAttr::get(dispatchOp.getContext(), bindingAttrs));
}

// Adds the entry point ops with assigned ordinals for each entry function.
// The entry points will all use the provided |interfaceOp|.
static LogicalResult declareEntryPointOps(
    IREE::Stream::ExecutableOp sourceExecutableOp,
    IREE::HAL::ExecutableOp targetExecutableOp,
    const BindingLayoutAnalysis &layoutAnalysis) {
  auto variantOps =
      targetExecutableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>();
  OpBuilder executableBuilder(&targetExecutableOp.getBlock().front());

  // For each exported function create a HAL entry point and dispatch thunk.
  int nextOrdinal = 0;
  for (auto exportOp :
       sourceExecutableOp.body().getOps<IREE::Stream::ExecutableExportOp>()) {
    int ordinal = nextOrdinal++;
    auto sourceFuncOp =
        sourceExecutableOp.getInnerModule().lookupSymbol<mlir::FuncOp>(
            exportOp.function_ref());
    if (failed(verifyEntryPointTypes(sourceFuncOp))) return failure();

    const auto &executableLayout = layoutAnalysis.getExecutableLayout(exportOp);

    // Create the interface for this entry point based on the analysis of its
    // usage within the program.
    auto layoutAttr =
        makeExecutableLayoutAttr(executableLayout, executableBuilder);

    // Clone the source function and update it to use the new interface.
    auto baseFuncOp =
        cloneFuncWithInterface(sourceFuncOp, executableLayout, layoutAttr);

    // Clone the updated function into each variant.
    for (auto variantOp : variantOps) {
      // Declare the entry point on the target.
      OpBuilder targetBuilder(&variantOp.getBlock().front());
      targetBuilder.create<IREE::HAL::ExecutableEntryPointOp>(
          exportOp.getLoc(),
          targetBuilder.getStringAttr(exportOp.function_ref()),
          targetBuilder.getIndexAttr(ordinal), layoutAttr, ArrayAttr{},
          IntegerAttr{});

      // Clone the updated interface-based function into the target.
      auto targetFuncOp = baseFuncOp.clone();
      variantOp.getInnerModule().push_back(targetFuncOp);
    }

    // Update all dispatch sites with the binding information.
    for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
      annotateDispatchSite(dispatchOp, executableLayout);
    }

    baseFuncOp.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.* info op conversion
//===----------------------------------------------------------------------===//

namespace {

template <typename SrcOp, typename DstOp>
struct ConvertDispatchWorkgroupInfoPattern final
    : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getResult().getType(),
                                       op.dimensionAttr());
    return success();
  }
};

struct InlineConstantWorkgroupSizePattern
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
  using OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp sizeOp,
                                PatternRewriter &rewriter) const override {
    // Lookup the entry point matching the parent.
    auto funcOp = sizeOp->getParentOfType<mlir::FuncOp>();
    auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    auto entryPointOp = dyn_cast<IREE::HAL::ExecutableEntryPointOp>(
        SymbolTable::lookupSymbolIn(variantOp, funcOp.getName()));
    assert(entryPointOp &&
           "must have an entry point corresponding to the parent func");
    auto workgroupSizeAttr = entryPointOp.workgroup_sizeAttr();
    if (!workgroupSizeAttr) return failure();

    uint64_t dimIdx = sizeOp.dimension().getZExtValue();
    auto dimAttr = workgroupSizeAttr[dimIdx];
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(sizeOp, dimAttr,
                                                   rewriter.getIndexType());
    return success();
  }
};

}  // namespace

static LogicalResult convertFlowInfoOps(IREE::HAL::ExecutableOp executableOp) {
  OwningRewritePatternList patterns(executableOp.getContext());
  patterns.insert<
      ConvertDispatchWorkgroupInfoPattern<IREE::Flow::DispatchWorkgroupIDOp,
                                          IREE::HAL::InterfaceWorkgroupIDOp>,
      ConvertDispatchWorkgroupInfoPattern<IREE::Flow::DispatchWorkgroupCountOp,
                                          IREE::HAL::InterfaceWorkgroupCountOp>,
      ConvertDispatchWorkgroupInfoPattern<IREE::Flow::DispatchWorkgroupSizeOp,
                                          IREE::HAL::InterfaceWorkgroupSizeOp>,
      InlineConstantWorkgroupSizePattern>(executableOp.getContext());
  return applyPatternsAndFoldGreedily(executableOp, std::move(patterns));
}

//===----------------------------------------------------------------------===//
// -iree-hal-materialize-interfaces
//===----------------------------------------------------------------------===//

class MaterializeInterfacesPass
    : public PassWrapper<MaterializeInterfacesPass, OperationPass<ModuleOp>> {
 public:
  MaterializeInterfacesPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-materialize-interfaces";
  }

  StringRef getDescription() const override {
    return "Materializes hal.executable ops from stream.executable ops";
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

    // Handle any hand-authored executables; these only need variant expansion
    // and no layout analysis as the user specified the layout themselves.
    if (failed(materializeExecutablesFromSourceOps(getOperation()))) {
      return signalPassFailure();
    }

    const auto &layoutAnalysis = getAnalysis<BindingLayoutAnalysis>();

    // Processes all executables within the input module and produce the
    // output HAL ops. We should ensure all deduping is performed prior to
    // this when it's easier to diff IR and where we still have the flow
    // context.
    auto sourceOps = llvm::to_vector<32>(
        getOperation().getOps<IREE::Stream::ExecutableOp>());
    for (auto sourceOp : sourceOps) {
      auto exportOps = sourceOp.getOps<IREE::Stream::ExecutableExportOp>();
      if (exportOps.empty()) continue;

      // Gather a list of all #hal.executable.targets that we should produce
      // variants for.
      auto targetAttrs =
          IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(sourceOp);
      if (targetAttrs.empty()) {
        sourceOp.emitError()
            << "no executable targets specified for translation";
        return signalPassFailure();
      }

      // Create the op that will contain the translated executable.
      OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
      builder.setInsertionPointAfter(sourceOp);
      auto executableOp = builder.create<IREE::HAL::ExecutableOp>(
          sourceOp.getLoc(), sourceOp.getName());
      executableOp.setVisibility(sourceOp.getVisibility());

      // Materialize all of the hal.executable.variant ops for all backends we
      // are targeting.
      SymbolTable targetSymbolTable(executableOp);
      OpBuilder targetBuilder(&executableOp.getBlock().back());
      for (auto targetAttr : targetAttrs) {
        auto targetContainerOp =
            targetBuilder.create<IREE::HAL::ExecutableVariantOp>(
                sourceOp->getLoc(), targetAttr.getSymbolNameFragment(),
                targetAttr);
        targetSymbolTable.insert(targetContainerOp);
        OpBuilder containerBuilder(&targetContainerOp.getBlock().back());
        containerBuilder.create<mlir::ModuleOp>(sourceOp->getLoc());
      }

      // Define interfaces for each exported function based on analysis.
      if (failed(
              declareEntryPointOps(sourceOp, executableOp, layoutAnalysis))) {
        return signalPassFailure();
      }

      // Convert interface-related flow.dispatch.* ops to their hal.interface.*
      // versions.
      if (failed(convertFlowInfoOps(executableOp))) {
        return signalPassFailure();
      }

      sourceOp.erase();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeInterfacesPass() {
  return std::make_unique<MaterializeInterfacesPass>();
}

static PassRegistration<MaterializeInterfacesPass> pass([] {
  return std::make_unique<MaterializeInterfacesPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
