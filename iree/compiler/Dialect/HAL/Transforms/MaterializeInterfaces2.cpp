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
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
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
// hal.executable.variant creation
//===----------------------------------------------------------------------===//

// Creates zero or more hal.executable.variant ops for each target backend.
// The source op will contain the flow.executable contents and any attributes
// the backend wants to carry along during transformation.
static LogicalResult declareVariantOps(IREE::Stream::ExecutableOp sourceOp,
                                       IREE::HAL::ExecutableOp executableOp) {
  // Gather a list of all #hal.executable.targets that we should produce
  // variants for.
  auto executableTargetAttrs =
      IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(sourceOp);
  if (executableTargetAttrs.empty()) {
    return sourceOp.emitError()
           << "no executable targets specified for translation";
  }

  // Materialize all of the hal.executable.variant ops for all backends we are
  // targeting. Note that each backend may create zero or more target ops.
  SymbolTable targetSymbolTable(executableOp);
  OpBuilder targetBuilder(&executableOp.getBlock().back());
  for (auto &targetAttr : executableTargetAttrs) {
    auto targetContainerOp =
        targetBuilder.create<IREE::HAL::ExecutableVariantOp>(
            sourceOp.getLoc(), targetAttr.getSymbolNameFragment(), targetAttr);
    targetSymbolTable.insert(targetContainerOp);
    OpBuilder containerBuilder(&targetContainerOp.getBlock().back());
    containerBuilder.create<mlir::ModuleOp>(sourceOp.getLoc());
  }

  // Ensure that at least one target op got created. If it didn't that means
  // the executable cannot be translated and it's better to fail now.
  if (executableOp.getBlock()
          .getOps<IREE::HAL::ExecutableVariantOp>()
          .empty()) {
    auto diagnostic = sourceOp.emitError();
    diagnostic
        << "no target backend was able to handle this executable; tried = [ ";
    for (const auto &targetAttr : executableTargetAttrs) {
      diagnostic << targetAttr.getFormat() << " ";
    }
    diagnostic << "]";
    return diagnostic;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Interface definition
//===----------------------------------------------------------------------===//

// Verifies that all types used with the given entry point are supportable.
static LogicalResult verifyEntryPointTypes(mlir::FuncOp entryFuncOp) {
  for (auto inputType : llvm::enumerate(entryFuncOp.getType().getInputs())) {
    if (inputType.value().isa<IREE::Stream::BindingType>()) {
      // OK - directly translates to a HAL interface binding.
    } else if (inputType.value().isa<IndexType>()) {
      // Index types are converted to platform bit-width later on.
      // TODO(benvanik): pick something here that the target devices support.
    } else if (auto integerType = inputType.value().dyn_cast<IntegerType>()) {
      if (integerType.getIntOrFloatBitWidth() != 32) {
        return entryFuncOp.emitError()
               << "unsupported argument " << inputType.index() << " bit depth "
               << integerType.getIntOrFloatBitWidth() << " (" << integerType
               << "); only 32-bit values are supported right now";
      }
    } else {
      return entryFuncOp.emitError()
             << "unsupported interface function argument " << inputType.index()
             << " type " << inputType.value()
             << "; requires tensors or simple primitive values (i32, etc)";
    }
  }
  return success();
}

struct Interface {
  // Materialized interface op with binding symbols.
  IREE::HAL::InterfaceOp op;
  // 1:1 with the function signature bindings. May be a subset of the interface.
  SmallVector<IREE::HAL::InterfaceBindingOp> resourceBindings;
};

// Creates an interface from an executable layout provided from analysis.
static Interface createInterface(Location loc,
                                 const ExecutableLayout &executableLayout,
                                 OpBuilder &executableBuilder) {
  Interface interface;
  interface.op = executableBuilder.create<IREE::HAL::InterfaceOp>(loc, "io");
  interface.op.push_constantsAttr(
      executableBuilder.getIndexAttr(executableLayout.pushConstantCount));
  auto interfaceBuilder = OpBuilder::atBlockBegin(&interface.op.body().front());
  DenseMap<std::pair<unsigned, unsigned>, IREE::HAL::InterfaceBindingOp>
      bindingMap;
  for (const auto &setLayout : executableLayout.setLayouts) {
    for (const auto &binding : setLayout.bindings) {
      std::string bindingName = "s" + std::to_string(setLayout.ordinal) + "b" +
                                std::to_string(binding.ordinal);
      if (allEnumBitsSet(binding.access,
                         IREE::HAL::MemoryAccessBitfield::Read |
                             IREE::HAL::MemoryAccessBitfield::Write)) {
        bindingName += "_rw";
      } else if (allEnumBitsSet(binding.access,
                                IREE::HAL::MemoryAccessBitfield::Read)) {
        bindingName += "_ro";
      } else if (allEnumBitsSet(binding.access,
                                IREE::HAL::MemoryAccessBitfield::Discard |
                                    IREE::HAL::MemoryAccessBitfield::Write)) {
        bindingName += "_xw";
      } else if (allEnumBitsSet(binding.access,
                                IREE::HAL::MemoryAccessBitfield::Write)) {
        bindingName += "_wo";
      }
      auto bindingOp = interfaceBuilder.create<IREE::HAL::InterfaceBindingOp>(
          interface.op.getLoc(), bindingName,
          /*set=*/APInt(64, setLayout.ordinal),
          /*binding=*/APInt(64, binding.ordinal), binding.type, binding.access);
      bindingMap.insert(
          {std::make_pair(setLayout.ordinal, binding.ordinal), bindingOp});
    }
  }
  for (auto setBinding : executableLayout.resourceMap) {
    interface.resourceBindings.push_back(bindingMap[setBinding]);
  }
  return interface;
}

// Converts the usage of the given primitive |arg| to interface methods.
static void convertOperandUsage(BlockArgument arg, unsigned pushConstantIdx,
                                OpBuilder &builder) {
  auto loadOp = builder.create<IREE::HAL::InterfaceLoadConstantOp>(
      arg.getLoc(), arg.getType(), APInt(64, pushConstantIdx));
  arg.replaceAllUsesWith(loadOp);
}

// Converts the usage of the given !stream.binding |arg| to interface methods.
static void convertBindingUsage(BlockArgument arg,
                                IREE::HAL::InterfaceOp interfaceOp,
                                IREE::HAL::InterfaceBindingOp bindingOp) {
  if (arg.use_empty()) return;  // no-op
  for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
    auto oldOp = dyn_cast<IREE::Stream::BindingSubspanOp>(use.getOwner());
    assert(oldOp && "bindings are only usable by stream.binding.subspan");
    OpBuilder builder(oldOp);
    auto newOp = builder.create<IREE::HAL::InterfaceBindingSubspanOp>(
        oldOp.getLoc(), oldOp.getType(),
        SymbolRefAttr::get(interfaceOp.sym_nameAttr(),
                           {SymbolRefAttr::get(bindingOp)}),
        oldOp.byte_offset(), /*byte_length=*/Value{}, oldOp.dynamic_dims());
    oldOp.replaceAllUsesWith(newOp.result());
    oldOp.erase();
  }
}

// Clones |sourceFuncOp| and updates its signature to match the |interfaceOp|
// and use the HAL interface access primitives.
static mlir::FuncOp cloneFuncWithInterface(
    mlir::FuncOp sourceFuncOp, const ExecutableLayout &executableLayout,
    Interface &interface) {
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
      // TODO(benvanik): symbolic push constant indices.
      convertOperandUsage(arg, operandIdx++, entryBuilder);
    }
  }
  unsigned bindingIdx = 0;
  for (auto arg : entryBlock->getArguments()) {
    if (arg.getType().isa<IREE::Stream::BindingType>()) {
      convertBindingUsage(arg, interface.op,
                          interface.resourceBindings[bindingIdx++]);
    }
  }

  // Remove all arguments now that we've turned them into lookup ops.
  entryBlock->eraseArguments([](auto arg) { return true; });

  return clonedFuncOp;
}

// Annotates |dispatchOp| with resource binding to interface binding mappings.
// TODO(benvanik): have a HAL op with structured information instead.
static void annotateDispatchSite(IREE::Stream::CmdDispatchOp dispatchOp,
                                 Interface &interface) {
  SmallVector<Attribute> bindingSymbols;
  for (auto resourceBinding : interface.resourceBindings) {
    bindingSymbols.push_back(
        SymbolRefAttr::get(dispatchOp.entry_pointAttr().getRootReference(),
                           {SymbolRefAttr::get(interface.op),
                            SymbolRefAttr::get(resourceBinding)}));
  }
  dispatchOp->setAttr("hal.interface.bindings",
                      ArrayAttr::get(dispatchOp.getContext(), bindingSymbols));
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
    auto interface = createInterface(sourceFuncOp.getLoc(), executableLayout,
                                     executableBuilder);

    // Clone the source function and update it to use the new interface.
    auto baseFuncOp =
        cloneFuncWithInterface(sourceFuncOp, executableLayout, interface);

    // Clone the updated function into each variant.
    for (auto variantOp : variantOps) {
      // Declare the entry point on the target.
      OpBuilder targetBuilder(&variantOp.getBlock().front());
      targetBuilder.create<IREE::HAL::ExecutableEntryPointOp>(
          exportOp.getLoc(),
          targetBuilder.getStringAttr(exportOp.function_ref()),
          targetBuilder.getIndexAttr(ordinal), SymbolRefAttr::get(interface.op),
          ArrayAttr{}, IntegerAttr{});

      // Clone the updated interface-based function into the target.
      auto targetFuncOp = baseFuncOp.clone();
      variantOp.getInnerModule().push_back(targetFuncOp);

      // Copy interface bindings into the target module so symbol references
      // work.
      auto inlinedInterfaceOp = interface.op.clone();
      inlinedInterfaceOp.setPrivate();
      variantOp.getInnerModule().push_back(inlinedInterfaceOp);
    }

    // Update all dispatch sites with the binding information.
    for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
      annotateDispatchSite(dispatchOp, interface);
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
// -iree-hal-materialize-interfaces2
//===----------------------------------------------------------------------===//

class MaterializeInterfaces2Pass
    : public PassWrapper<MaterializeInterfaces2Pass, OperationPass<ModuleOp>> {
 public:
  MaterializeInterfaces2Pass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-materialize-interfaces2";
  }

  StringRef getDescription() const override {
    return "Materializes hal.executable ops from stream.executable ops";
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

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

      // Create the op that will contain the translated executable.
      OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
      builder.setInsertionPointAfter(sourceOp);
      auto executableOp = builder.create<IREE::HAL::ExecutableOp>(
          sourceOp.getLoc(), sourceOp.getName());
      executableOp.setVisibility(sourceOp.getVisibility());

      // Embed the hal.executable.variant ops for each source.
      if (failed(declareVariantOps(sourceOp, executableOp))) {
        return signalPassFailure();
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

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeInterfaces2Pass() {
  return std::make_unique<MaterializeInterfaces2Pass>();
}

static PassRegistration<MaterializeInterfaces2Pass> pass([] {
  return std::make_unique<MaterializeInterfaces2Pass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
