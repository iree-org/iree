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

#include "iree/compiler/Dialect/HAL/Target/VMLA/VMLATarget.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"
#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/vmla_executable_def_builder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

// Destructively merges |sourceModuleOp| into |targetModuleOp|.
// |targetSymbolTable| is updated with the new symbols.
void mergeModuleInto(IREE::VM::ModuleOp sourceModuleOp,
                     IREE::VM::ModuleOp targetModuleOp,
                     DenseMap<StringRef, Operation *> &targetSymbolMap) {
  auto allOps = llvm::to_vector<8>(llvm::map_range(
      sourceModuleOp.getBlock(), [&](Operation &op) { return &op; }));
  for (auto &op : allOps) {
    if (op->isKnownTerminator()) continue;
    if (auto symbolInterface = dyn_cast<SymbolOpInterface>(op)) {
      if (targetSymbolMap.count(symbolInterface.getName())) {
        // TODO(scotttodd): compare ops to ensure we aren't copying different
        // things with the same name.
        continue;
      }
      targetSymbolMap[symbolInterface.getName()] = op;
    }
    op->moveBefore(&targetModuleOp.getBlock().back());
  }

  // Now that we're done cloning its ops, delete the original target op.
  sourceModuleOp.erase();
}

// Replaces each usage of an entry point with its original symbol name with a
// new symbol name.
void replaceEntryPointUses(mlir::ModuleOp moduleOp,
                           const DenseMap<Attribute, Attribute> &replacements) {
  for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
    funcOp.walk([&](IREE::HAL::CommandBufferDispatchSymbolOp dispatchOp) {
      auto it = replacements.find(dispatchOp.entry_point());
      if (it != replacements.end()) {
        dispatchOp.entry_pointAttr(it->second.cast<SymbolRefAttr>());
      }
    });
  }
}

}  // namespace

VMLATargetOptions getVMLATargetOptionsFromFlags() {
  VMLATargetOptions targetOptions;
  // TODO(benvanik): flags.
  return targetOptions;
}

class VMLATargetBackend final : public TargetBackend {
 public:
  VMLATargetBackend(VMLATargetOptions options) : options_(std::move(options)) {}

  std::string name() const override { return "vmla"; }
  std::string filter_pattern() const override { return "vmla"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VM::VMDialect, VMLA::VMLADialect>();
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    IREE::VMLA::buildVMLATransformPassPipeline(passManager);

    // TODO(#614): remove this when the std->vm conversion isn't looking for
    // iree.module.export.
    passManager.addPass(IREE::VM::createMarkPublicSymbolsExportedPass());

    IREE::VM::buildVMTransformPassPipeline(
        passManager, IREE::VM::getTargetOptionsFromFlags());
  }

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override {
    OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());
    auto executableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());

    // Create our new "linked" hal.executable.
    auto linkedExecutableOp = builder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), "linked_vmla");
    linkedExecutableOp.setPrivate();
    // Add our VMLA hal.executable.target with an empty module.
    builder.setInsertionPointToStart(linkedExecutableOp.getBody());
    auto linkedTargetOp = builder.create<IREE::HAL::ExecutableTargetOp>(
        moduleOp.getLoc(), name(), filter_pattern());
    builder.setInsertionPoint(&linkedTargetOp.getBlock().back());
    auto linkedModuleOp = builder.create<ModuleOp>(moduleOp.getLoc());
    // Add an empty vm.module to that module.
    builder.setInsertionPointToStart(linkedModuleOp.getBody());
    auto linkedVmModuleOp =
        builder.create<IREE::VM::ModuleOp>(moduleOp.getLoc(), "linked_module");

    llvm::SmallVector<IREE::HAL::InterfaceOp, 4> interfaceOps;
    int nextEntryPointOrdinal = 0;
    DenseMap<StringRef, Operation *> symbolMap;
    DenseMap<Attribute, Attribute> entryPointRefReplacements;
    auto linkedExecutableBuilder =
        OpBuilder::atBlockBegin(linkedExecutableOp.getBody());
    auto linkedTargetBuilder =
        OpBuilder::atBlockBegin(linkedTargetOp.getBody());
    for (auto executableOp : executableOps) {
      auto targetOps = llvm::to_vector<4>(
          executableOp.getOps<IREE::HAL::ExecutableTargetOp>());
      for (auto targetOp : targetOps) {
        // Only process targets matching our pattern.
        if (!matchPattern(targetOp.target_backend_filter(), filter_pattern())) {
          continue;
        }

        IREE::HAL::InterfaceOp interfaceOpForExecutable;
        for (auto interfaceOp : interfaceOps) {
          if (interfaceOp.isEquivalentTo(executableOp.getFirstInterfaceOp())) {
            interfaceOpForExecutable = interfaceOp;
            break;
          }
        }
        if (!interfaceOpForExecutable) {
          interfaceOpForExecutable =
              dyn_cast<IREE::HAL::InterfaceOp>(linkedExecutableBuilder.clone(
                  *executableOp.getFirstInterfaceOp()));
          interfaceOpForExecutable.setName(
              llvm::formatv("legacy_io_{0}", interfaceOps.size()).str());
          interfaceOps.push_back(interfaceOpForExecutable);
        }

        // Clone entry point ops and queue remapping ordinals and updating
        // symbol refs.
        for (auto entryPointOp :
             targetOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
          auto newEntryPointOp =
              linkedTargetBuilder.create<IREE::HAL::ExecutableEntryPointOp>(
                  entryPointOp.getLoc(), entryPointOp.sym_nameAttr(),
                  builder.getI32IntegerAttr(nextEntryPointOrdinal++),
                  builder.getSymbolRefAttr(interfaceOpForExecutable.getName()),
                  entryPointOp.signatureAttr(), ArrayAttr{});

          // Add to replacement table for fixing up dispatch calls referencing
          // this entry point.
          auto oldSymbolRefAttr = builder.getSymbolRefAttr(
              executableOp.getName(), {builder.getSymbolRefAttr(targetOp),
                                       builder.getSymbolRefAttr(entryPointOp)});
          auto newSymbolRefAttr = builder.getSymbolRefAttr(
              linkedExecutableOp.getName(),
              {builder.getSymbolRefAttr(linkedTargetOp),
               builder.getSymbolRefAttr(newEntryPointOp)});
          entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
        }

        // Merge the existing vm.module op into the new linked vm.module op.
        auto vmModuleOps =
            targetOp.getInnerModule().getOps<IREE::VM::ModuleOp>();
        if (vmModuleOps.empty()) {
          return targetOp.getInnerModule().emitError()
                 << "target's outer module does not contain a vm.module op";
        }
        mergeModuleInto(*vmModuleOps.begin(), linkedVmModuleOp, symbolMap);

        targetOp.erase();
      }

      if (executableOp.getOps<IREE::HAL::ExecutableTargetOp>().empty()) {
        executableOp.erase();
      }
    }

    // Update references to @executable::@target::@entry symbols.
    replaceEntryPointUses(moduleOp, entryPointRefReplacements);

    // Remove if we didn't add anything.
    if (linkedTargetOp.getOps<IREE::HAL::ExecutableEntryPointOp>().empty()) {
      linkedTargetOp.erase();
      linkedExecutableOp.erase();
    }

    return success();
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Serialize the VM module to bytes directly into a flatbuffer.
    FlatbufferBuilder builder;
    IREE::VM::BytecodeTargetOptions bytecodeOptions;
    auto dataRef = builder.streamUint8Vec([&](raw_ostream &stream) {
      return succeeded(translateModuleToBytecode(targetOp.getInnerModule(),
                                                 bytecodeOptions, stream));
    });
    if (!dataRef) {
      return targetOp.emitError() << "failed to serialize converted VM module";
    }

    // Pack the executable definition and get the bytes with the proper header.
    // The header is used to verify the contents at runtime.
    iree_VMLAExecutableDef_start_as_root(builder);
    iree_VMLAExecutableDef_bytecode_module_add(builder, dataRef);
    iree_VMLAExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    // NOTE: this snapshots the flatbuffer builder data at the time it is called
    // and future changes will not be observed.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::VMLA),
        builder.getBufferAttr(executableBuilder.getContext()));
    return success();
  }

  std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder) override {
    // For now we are not tiling and just dispatch everything as 1,1,1.
    auto constantOne = builder.createOrFold<mlir::ConstantIndexOp>(loc, 1);
    return {constantOne, constantOne, constantOne};
  }

 private:
  VMLATargetOptions options_;
};

void registerVMLATargetBackends(
    std::function<VMLATargetOptions()> queryOptions) {
  getVMLATargetOptionsFromFlags();
  static TargetBackendRegistration registration("vmla", [=]() {
    return std::make_unique<VMLATargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
