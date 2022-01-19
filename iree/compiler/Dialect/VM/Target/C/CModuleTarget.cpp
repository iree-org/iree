// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/DropExcludedExports.h"
#include "iree/compiler/Dialect/VM/Target/C/CppEmitter.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

static void printCompilerConfigurationBlock(llvm::raw_ostream &output) {
  output << "//" << std::string(77, '=') << "\n"
         << "// compiler configuration\n"
         << "//" << std::string(77, '=') << "\n\n";
}

static void printModuleComment(IREE::VM::ModuleOp &moduleOp,
                               llvm::raw_ostream &output) {
  output << "//" << std::string(77, '=') << "\n"
         << "// module \"" << moduleOp.getName()
         << "\"\n"
            "//"
         << std::string(77, '=') << "\n";
}

static LogicalResult printRodataBuffers(IREE::VM::ModuleOp &moduleOp,
                                        mlir::emitc::CppEmitter &emitter) {
  llvm::raw_ostream &output = emitter.ostream();
  std::string moduleName = moduleOp.getName().str();

  for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
    auto value =
        rodataOp.value().dyn_cast<IREE::Util::SerializableAttrInterface>();
    assert(value && "expected a serializable rodata value");
    SmallVector<char> byteBuffer;
    if (failed(value.serializeToVector(llvm::support::endianness::little,
                                       byteBuffer))) {
      return rodataOp.emitError() << "error during serialization";
    }

    constexpr size_t kDefaultRodataAlignment = 16;
    size_t alignment =
        rodataOp.alignment()
            ? static_cast<size_t>(rodataOp.alignment().getValue())
            : 0;
    if (alignment == 0) alignment = kDefaultRodataAlignment;

    std::string bufferName =
        moduleOp.getName().str() + "_" + rodataOp.getName().str();

    output << "iree_alignas(" << alignment << ") static const uint8_t "
           << bufferName << "[] = {";
    llvm::interleaveComma(byteBuffer, output, [&](char value) {
      output << static_cast<unsigned int>(static_cast<unsigned char>(value));
    });
    output << "};\n";
  }

  output << "\n";

  return success();
}

static LogicalResult printStructDefinitions(IREE::VM::ModuleOp &moduleOp,
                                            mlir::emitc::CppEmitter &emitter) {
  llvm::raw_ostream &output = emitter.ostream();
  std::string moduleName = moduleOp.getName().str();

  output << "struct " << moduleName << "_t {\n";
  output << "iree_allocator_t allocator;\n";
  output << "};\n";

  output << "struct " << moduleName << "_state_t {\n";

  // Returns |count| or 1 if |count| == 0.
  // Some compilers (MSVC) don't support zero-length struct fields on the
  // interior of structs (just VLA at the tail).
  auto countOrEmpty = [](uint32_t count) { return count ? count : 1; };

  auto ordinalCounts = moduleOp.ordinal_counts().getValue();
  output << "iree_allocator_t allocator;\n";
  output << "uint8_t rwdata[" << countOrEmpty(ordinalCounts.global_bytes())
         << "];\n";
  output << "iree_vm_ref_t refs[" << countOrEmpty(ordinalCounts.global_refs())
         << "];\n";
  output << "iree_vm_buffer_t rodata_buffers["
         << countOrEmpty(ordinalCounts.rodatas()) << "];\n";
  output << "iree_vm_function_t imports["
         << countOrEmpty(ordinalCounts.import_funcs()) << "];\n";
  output << "};\n";

  output << "typedef struct " << moduleName << "_t " << moduleName << "_t;\n";
  output << "typedef struct " << moduleName << "_state_t " << moduleName
         << "_state_t;\n";

  output << "\n";

  return success();
}

static LogicalResult buildModuleDescriptors(IREE::VM::ModuleOp &moduleOp,
                                            mlir::emitc::CppEmitter &emitter) {
  SymbolTable symbolTable(moduleOp);
  std::string moduleName = moduleOp.getName().str();
  llvm::raw_ostream &output = emitter.ostream();

  auto printStringView = [](StringRef s) -> std::string {
    // We can't use iree_make_string_view because function calls are not allowed
    // for constant expressions in C.
    // TODO(#7605): Switch to IREE_SVL. We can't use IREE_SVL today because it
    // uses designated initializers, which cause issues when compiled as C++.
    return ("{\"" + s + "\", " + std::to_string(s.size()) + "}").str();
  };

  // exports
  SmallVector<IREE::VM::ExportOp, 4> exportOps(
      moduleOp.getOps<IREE::VM::ExportOp>());
  std::string exportName = moduleName + "_exports_";
  output << "static const iree_vm_native_export_descriptor_t " << exportName
         << "[] = {\n";
  if (exportOps.empty()) {
    // Empty list placeholder.
    output << "    {0},\n";
  } else {
    // sort export ops
    llvm::sort(exportOps, [](auto &lhs, auto &rhs) {
      return lhs.export_name().compare(rhs.export_name()) < 0;
    });
    for (auto exportOp : exportOps) {
      StringRef funcName = exportOp.function_ref();
      auto funcOp = symbolTable.lookup<mlir::FuncOp>(funcName);
      if (!funcOp) {
        return exportOp.emitError("Couldn't find referenced FuncOp");
      }
      StringAttr callingConvention = funcOp.getOperation()
                                         ->getAttr("vm.calling_convention")
                                         .cast<StringAttr>();
      if (!callingConvention) {
        return exportOp.emitError("Couldn't find calling convention attribute");
      }

      // TODO(simon-camp): support function-level reflection attributes
      output << "{" << printStringView(exportOp.export_name()) << ", "
             << printStringView(callingConvention.getValue())
             << ", 0, NULL},\n";
    }
  }
  output << "};\n";
  output << "\n";

  // imports
  SmallVector<IREE::VM::ImportOp, 4> importOps(
      moduleOp.getOps<IREE::VM::ImportOp>());
  std::string importName = moduleName + "_imports_";
  output << "static const iree_vm_native_import_descriptor_t " << importName
         << "[] = {\n";
  if (importOps.empty()) {
    // Empty list placeholder.
    output << "    {0},\n";
  } else {
    // sort import ops by ordinal
    llvm::sort(importOps, [](auto &lhs, auto &rhs) {
      return lhs.ordinal().getValue().getZExtValue() <
             rhs.ordinal().getValue().getZExtValue();
    });
    for (auto importOp : importOps) {
      output << "{" << printStringView(importOp.getName()) << "},\n";
    }
  }
  output << "};\n";
  output << "\n";

  // functions
  std::string functionName = moduleName + "_funcs_";
  output << "static const iree_vm_native_function_ptr_t " << functionName
         << "[] = {\n";
  if (exportOps.empty()) {
    // Empty list placeholder.
    output << "    {0},\n";
  } else {
    // We only add exported functions to the table, as calls to internal
    // functions are directly mapped to C function calls of the generated
    // implementation.
    for (auto exportOp : exportOps) {
      StringRef funcName = exportOp.function_ref();
      auto funcOp = symbolTable.lookup<mlir::FuncOp>(funcName);
      if (!funcOp) {
        return exportOp.emitError("Couldn't find referenced FuncOp");
      }
      output << "{"
             << "(iree_vm_native_function_shim_t)iree_emitc_shim, "
             << "(iree_vm_native_function_target_t)" << funcName << "},\n";
    }
  }
  output << "};\n";
  output << "\n";

  // module descriptor
  // TODO(simon-camp): support module-level reflection attributes
  std::string descriptorName = moduleName + "_descriptor_";
  output << "static const iree_vm_native_module_descriptor_t " << descriptorName
         << " = {\n"
         << printStringView(moduleName) << ",\n"
         << importOps.size() << ",\n"
         << importName << ",\n"
         << exportOps.size() << ",\n"
         << exportName << ",\n"
         << exportOps.size() << ",\n"
         << functionName << ",\n"
         << "0,\n"
         << "NULL,\n"
         << "};\n";

  output << "\n";
  return success();
}

/// Adapted from BytecodeModuleTarget and extended by C specific passes
static LogicalResult canonicalizeModule(
    IREE::VM::ModuleOp moduleOp, IREE::VM::CTargetOptions targetOptions) {
  OwningRewritePatternList patterns(moduleOp.getContext());
  ConversionTarget target(*moduleOp.getContext());
  target.addLegalDialect<IREE::VM::VMDialect>();
  target.addLegalOp<IREE::Util::DoNotOptimizeOp>();

  // Add all VM canonicalization patterns and mark pseudo-ops illegal.
  auto *context = moduleOp.getContext();
  for (auto op : context->getRegisteredOperations()) {
    // Non-serializable ops must be removed prior to serialization.
    if (op.hasTrait<OpTrait::IREE::VM::PseudoOp>()) {
      op.getCanonicalizationPatterns(patterns, context);
      target.setOpAction(op, ConversionTarget::LegalizationAction::Illegal);
    }

    // Debug ops must not be present when stripping.
    // TODO(benvanik): add RemoveDisabledDebugOp pattern.
    if (op.hasTrait<OpTrait::IREE::VM::DebugOnly>() &&
        targetOptions.stripDebugOps) {
      target.setOpAction(op, ConversionTarget::LegalizationAction::Illegal);
    }
  }

  if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
    return moduleOp.emitError() << "unable to fully apply conversion to module";
  }

  PassManager passManager(context);
  mlir::applyPassManagerCLOptions(passManager);
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  auto &modulePasses = passManager.nest<IREE::VM::ModuleOp>();

  if (targetOptions.optimize) {
    // TODO(benvanik): does this run until it quiesces?
    modulePasses.addPass(mlir::createInlinerPass());
    modulePasses.addPass(mlir::createCSEPass());
    modulePasses.addPass(mlir::createCanonicalizerPass());
  }

  // C target specific pass

  // Erase exports annotated with 'emitc.exclude'. This makes testing
  // of partially supported ops easier. For the DCE pass to remove the
  // referenced function it must be unused and marked private.
  modulePasses.addPass(createDropExcludedExportsPass());
  modulePasses.addPass(mlir::createSymbolDCEPass());

  // In the the Bytecode module the order is:
  // * `createDropCompilerHintsPass()`
  // * `IREE::VM::createOrdinalAllocationPass()`
  // Here, we have to reverse the order and run
  // `createConvertVMToEmitCPass()` inbetween to test the EmitC pass.
  // Otherwise, the constants get folded by the canonicalizer.

  // Mark up the module with ordinals for each top-level op (func, etc).
  // This will make it easier to correlate the MLIR textual output to the
  // binary output.
  // We don't want any more modifications after this point as they could
  // invalidate the ordinals.
  modulePasses.addPass(IREE::VM::createOrdinalAllocationPass());

  // C target specific pass
  modulePasses.addPass(createConvertVMToEmitCPass());

  modulePasses.addPass(IREE::Util::createDropCompilerHintsPass());

  if (failed(passManager.run(moduleOp->getParentOfType<mlir::ModuleOp>()))) {
    return moduleOp.emitError() << "failed during transform passes";
  }

  return success();
}

LogicalResult translateModuleToC(IREE::VM::ModuleOp moduleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output) {
  moduleOp.getContext()->getOrLoadDialect<IREE::Util::UtilDialect>();

  if (failed(canonicalizeModule(moduleOp, targetOptions))) {
    return moduleOp.emitError()
           << "failed to canonicalize vm.module to a serializable form";
  }

  if (targetOptions.outputFormat == COutputFormat::kMlirText) {
    // Use the standard MLIR text printer.
    moduleOp.getOperation()->print(output);
    output << "\n";
    return success();
  }

  auto printInclude = [&output](std::string include) {
    output << "#include \"" << include << "\"\n";
  };

  printInclude("iree/vm/api.h");
  printInclude("iree/vm/ops.h");
  printInclude("iree/vm/ops_emitc.h");
  printInclude("iree/vm/shims_emitc.h");
  output << "\n";

  printCompilerConfigurationBlock(output);
  output << "\n";

  printModuleComment(moduleOp, output);
  output << "\n";

  mlir::emitc::CppEmitter emitter(output, /*declareVariablesAtTop=*/true);
  mlir::emitc::CppEmitter::Scope scope(emitter);

  if (failed(printRodataBuffers(moduleOp, emitter))) {
    return failure();
  }

  // build struct definitions
  if (failed(printStructDefinitions(moduleOp, emitter))) {
    return failure();
  }

  // translate functions
  output << "// DECLARE FUNCTIONS\n";

  for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
    Operation *op = funcOp.getOperation();
    if (op->hasAttr("emitc.static")) output << "static ";

    if (failed(
            emitter.emitTypes(funcOp.getLoc(), funcOp.getType().getResults())))
      return failure();
    output << " " << funcOp.getName();

    output << "(";

    bool error = false;
    llvm::interleaveComma(
        funcOp.getArguments(), output, [&](BlockArgument arg) {
          if (failed(emitter.emitType(funcOp.getLoc(), arg.getType())))
            error = true;
        });
    if (error) return failure();
    output << ");\n";
  }

  output << "// DEFINE FUNCTIONS\n";

  // Emit code for functions skipping those marked with `vm.emit_at_end`.
  for (Operation &op : moduleOp.getOps()) {
    // TODO(simon-camp): Clean up. We generate calls to a macro that defines a
    // struct. As we declare all variables at the start of the function, the
    // macro call cannot be inlined into the function.
    if (!isa<mlir::FuncOp, emitc::CallOp>(op)) continue;
    if (op.hasAttr("vm.emit_at_end")) continue;
    if (op.hasAttr("emitc.static")) output << "static ";
    if (failed(emitter.emitOperation(op,
                                     /*trailingSemicolon=*/false)))
      return failure();
  }

  output << "\n";

  // generate module descriptors
  if (failed(buildModuleDescriptors(moduleOp, emitter))) {
    return failure();
  }

  // Emit code for functions marked with `vm.emit_at_end`.
  for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
    Operation *op = funcOp.getOperation();
    if (!op->hasAttr("vm.emit_at_end")) continue;
    if (op->hasAttr("emitc.static")) output << "static ";
    if (failed(emitter.emitOperation(*funcOp.getOperation(),
                                     /*trailingSemicolon=*/false)))
      return failure();
  }

  return success();
}

LogicalResult translateModuleToC(mlir::ModuleOp outerModuleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output) {
  auto moduleOps = outerModuleOp.getOps<IREE::VM::ModuleOp>();
  if (moduleOps.empty()) {
    return outerModuleOp.emitError()
           << "outer module does not contain a vm.module op";
  }
  return translateModuleToC(*moduleOps.begin(), targetOptions, output);
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
