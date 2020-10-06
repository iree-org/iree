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

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"
#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"
#include "iree/schemas/vmla_executable_def_generated.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

bool AreInterfacesEquivalent(IREE::HAL::InterfaceOp lhs,
                             IREE::HAL::InterfaceOp rhs) {
  auto lhsBindings = lhs.getBlock().getOps<IREE::HAL::InterfaceBindingOp>();
  auto rhsBindings = rhs.getBlock().getOps<IREE::HAL::InterfaceBindingOp>();
  auto lhsIt = lhsBindings.begin(), lhsEnd = lhsBindings.end();
  auto rhsIt = rhsBindings.begin(), rhsEnd = rhsBindings.end();
  for (; lhsIt != lhsEnd && rhsIt != rhsEnd; ++lhsIt, ++rhsIt) {
    // Assume bindings are in order, check equivalence of each pairing.
    if (!OperationEquivalence::isEquivalentTo(*lhsIt, *rhsIt)) return false;
  }

  if (lhsIt != lhsEnd || rhsIt != rhsEnd) {
    // Not finished iterating through one, number of interface bindings differ.
    return false;
  }

  return true;
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

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpPassManager &passManager) override {
    IREE::VMLA::buildVMLATransformPassPipeline(passManager);

    // TODO(#614): remove this when the std->vm conversion isn't looking for
    // iree.module.export.
    passManager.addPass(IREE::VM::createMarkPublicSymbolsExportedPass());

    IREE::VM::buildVMTransformPassPipeline(
        passManager, IREE::VM::getTargetOptionsFromFlags());
  }

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override {
    // --- Linking overview ---
    //
    // We start with a `module` containing multiple `hal.executable`s, each with
    // potentially multiple `hal.executable.target`s. We want to move all
    // compatible VMLA functions into a new "linked" executable, de-duping
    // symbols, and updating references as we go.
    //
    // Sample IR after:
    //   hal.executable @linked_vmla {
    //     hal.interface @legacy_io_0 { ... }
    //     hal.interface @legacy_io_1 { ... }
    //     hal.executable.target @vmla, filter="vmla" {
    //       hal.executable.entry_point @main_dispatch_0 attributes { ... }
    //       hal.executable.entry_point @main_dispatch_1 attributes { ... }
    //       hal.executable.entry_point @main_dispatch_2 attributes { ... }
    //       module {
    //         vm.module @module {
    //           vm.func @main_0(...) { ... }
    //           vm.func @main_1(...) { ... }
    //           vm.func @main_2(...) { ... }
    //         }
    //       }
    //     }
    //   }
    //   hal.executable @main_dispatch_0 {
    //     hal.interface @legacy_io { ... }
    //     hal.executable.target @other, filter="other" {
    //       hal.executable.entry_point @main_dispatch_0 attributes { ... }
    //       module { ... }
    //     }
    //   }

    OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());
    auto executableOps = moduleOp.getOps<IREE::HAL::ExecutableOp>();

    // Create our new "linked" hal.executable.
    auto linkedExecutableOp = builder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), "linked_vmla");
    SymbolTable::setSymbolVisibility(linkedExecutableOp,
                                     SymbolTable::Visibility::Private);
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

    int executablesLinked = 0;
    llvm::SmallVector<IREE::HAL::InterfaceOp, 4> interfaceOps;
    int nextEntryPointOrdinal = 0;
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
          if (AreInterfacesEquivalent(interfaceOp,
                                      executableOp.getFirstInterfaceOp())) {
            interfaceOpForExecutable = interfaceOp;
          }
        }
        if (!interfaceOpForExecutable) {
          builder.setInsertionPoint(linkedTargetOp);
          interfaceOpForExecutable = dyn_cast<IREE::HAL::InterfaceOp>(
              builder.clone(*executableOp.getFirstInterfaceOp()));
          interfaceOpForExecutable.setName(
              llvm::formatv("legacy_io_{0}", interfaceOps.size()).str());
          interfaceOps.push_back(interfaceOpForExecutable);
        }

        // Clone entry point ops, remapping ordinals and updating symbol refs.
        builder.setInsertionPoint(linkedModuleOp);
        for (auto entryPointOp :
             targetOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
          auto newEntryPointOp =
              builder.create<IREE::HAL::ExecutableEntryPointOp>(
                  entryPointOp.getLoc(), entryPointOp.sym_nameAttr(),
                  builder.getI32IntegerAttr(nextEntryPointOrdinal++),
                  builder.getSymbolRefAttr(interfaceOpForExecutable.getName()),
                  entryPointOp.signatureAttr());

          // Update references to @executable::@target::@entry symbols.
          // SymbolTable::replaceAllSymbolUses only looks at root symbols,
          // which we can't blindly replace (other targets will map to other
          // linked executables).
          auto executableUses =
              SymbolTable::getSymbolUses(executableOp, moduleOp);
          if (!executableUses.hasValue()) continue;
          for (auto executableUse : executableUses.getValue()) {
            auto executableUser = executableUse.getUser();
            // Only process symbols for this @target::@entry.
            auto nestedRefs =
                executableUse.getSymbolRef().getNestedReferences();
            if (nestedRefs.size() != 2 ||
                nestedRefs[0].getValue() != targetOp.sym_name() ||
                nestedRefs[1].getValue() != entryPointOp.sym_name()) {
              continue;
            }
            if (auto dispatchOp =
                    dyn_cast<IREE::HAL::CommandBufferDispatchSymbolOp>(
                        executableUser)) {
              // New nested reference to the linked exe/target/entry.
              StringRef newExecutableOpSymName =
                  linkedExecutableOp
                      .getAttrOfType<StringAttr>(
                          SymbolTable::getSymbolAttrName())
                      .getValue();
              auto newSymbolRefAttr = builder.getSymbolRefAttr(
                  newExecutableOpSymName,
                  {builder.getSymbolRefAttr(linkedTargetOp),
                   builder.getSymbolRefAttr(newEntryPointOp)});
              dispatchOp.setAttr("entry_point", newSymbolRefAttr);
            }
          }
        }

        // Clone vm.module ops, including their contents.
        auto vmModuleOps =
            targetOp.getInnerModule().getOps<IREE::VM::ModuleOp>();
        if (vmModuleOps.empty()) {
          return targetOp.getInnerModule().emitError()
                 << "target's outer module does not contain a vm.module op";
        }
        auto vmModuleOp = *vmModuleOps.begin();
        builder.setInsertionPoint(&linkedVmModuleOp.getBlock().back());
        // Use a SymbolTable to guard against inserting duplicate symbols.
        SymbolTable symbolTable(linkedVmModuleOp.getOperation());

        for (auto &op : vmModuleOp.getBody()->getOperations()) {
          if (auto terminatorOp = dyn_cast<IREE::VM::ModuleTerminatorOp>(op)) {
            continue;
          }
          if (op.hasTrait<SymbolOpInterface::Trait>() &&
              symbolTable.lookup(dyn_cast<SymbolOpInterface>(op).getName())) {
            continue;
          }
          builder.clone(op);
        }

        // Now that we're done cloning its ops, delete the original target op.
        targetOp.erase();

        executablesLinked++;
      }
    }

    if (executablesLinked == 0) {
      linkedExecutableOp.erase();
    }

    return success();
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Serialize the VM module to bytes.
    std::string byteStreamValue;
    llvm::raw_string_ostream byte_stream(byteStreamValue);
    IREE::VM::BytecodeTargetOptions bytecodeOptions;
    if (failed(translateModuleToBytecode(targetOp.getInnerModule(),
                                         bytecodeOptions, byte_stream))) {
      return targetOp.emitError() << "failed to serialize converted VM module";
    }

    // Pack the executable definition and get the bytes with the proper header.
    // The header is used to verify the contents at runtime.
    ::flatbuffers::FlatBufferBuilder fbb;
    iree::VMLAExecutableDefT vmlaExecutableDef;
    vmlaExecutableDef.bytecode_module.resize(byteStreamValue.size());
    std::memcpy(vmlaExecutableDef.bytecode_module.data(),
                byteStreamValue.data(), byteStreamValue.size());
    auto executableOffset =
        iree::VMLAExecutableDef::Pack(fbb, &vmlaExecutableDef);
    iree::FinishVMLAExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::VMLA),
        std::move(bytes));

    return success();
  }

  std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
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
