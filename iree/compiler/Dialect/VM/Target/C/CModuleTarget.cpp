// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"

#include "emitc/Target/Cpp/CppEmitter.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/DropExcludedExports.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Utils/ConstantEncoding.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

static void printModuleComment(IREE::VM::ModuleOp &moduleOp,
                               llvm::raw_ostream &output) {
  output << "//" << std::string(77, '=') << "\n"
         << "// module \"" << moduleOp.getName()
         << "\"\n"
            "//"
         << std::string(77, '=') << "\n";
}

static void printSeparatingComment(llvm::raw_ostream &output) {
  output << "//" << std::string(77, '=')
         << "\n"
            "// The code below setups functions and lookup tables to "
            "implement the vm\n"
            "// interface\n"
            "//"
         << std::string(77, '=') << "\n";
}
static LogicalResult printRodataBuffers(IREE::VM::ModuleOp &moduleOp,
                                        mlir::emitc::CppEmitter &emitter) {
  llvm::raw_ostream &output = emitter.ostream();
  std::string moduleName = moduleOp.getName().str();

  for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
    ElementsAttr value = rodataOp.value();
    auto bitwidth = value.getType().getElementTypeBitWidth();
    size_t size = value.getNumElements() * (bitwidth / 8);
    SmallVector<uint8_t, 32> byteBuffer;
    byteBuffer.resize(size);

    constexpr size_t kDefaultRodataAlignment = 16;

    size_t alignment =
        rodataOp.alignment()
            ? static_cast<size_t>(rodataOp.alignment().getValue())
            : 0;
    if (alignment == 0) alignment = kDefaultRodataAlignment;

    if (failed(serializeConstantArray(rodataOp.getLoc(), value, alignment,
                                      byteBuffer.data()))) {
      return rodataOp.emitError() << "error during serialization";
    }

    std::string buffer_name =
        moduleOp.getName().str() + "_" + rodataOp.getName().str();

    output << "iree_alignas(" << alignment << ") static const uint8_t "
           << buffer_name << "[] = {";
    llvm::interleaveComma(byteBuffer, output, [&](uint8_t value) {
      output << static_cast<unsigned int>(value);
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

  output << "struct " << moduleName << "_t {};\n";
  output << "struct " << moduleName << "_state_t {\n";

  output << "iree_allocator_t allocator;\n";
  output << "uint8_t rwdata["
         << moduleOp.ordinal_counts().getValue().global_bytes() << "];\n";
  output << "iree_vm_ref_t refs["
         << moduleOp.ordinal_counts().getValue().global_refs() << "];\n";
  output << "iree_vm_buffer_t rodata_buffers["
         << moduleOp.ordinal_counts().getValue().rodatas() << "];\n";
  output << "iree_vm_function_t imports["
         << moduleOp.ordinal_counts().getValue().import_funcs() << "];\n";
  output << "};\n";

  output << "typedef struct " << moduleName << "_t " << moduleName << "_t;\n";
  output << "typedef struct " << moduleName << "_state_t " << moduleName
         << "_state_t;\n";

  output << "\n";

  return success();
}

static LogicalResult printShim(mlir::FuncOp &funcOp,
                               llvm::raw_ostream &output) {
  StringAttr callingConvention =
      funcOp.getOperation()->getAttr("calling_convention").cast<StringAttr>();
  if (!callingConvention) {
    return funcOp.emitError("Couldn't create calling convention string");
  }
  output << "call_" << callingConvention.getValue() << "_shim";
  return success();
}

static LogicalResult initializeState(IREE::VM::ModuleOp moduleOp,
                                     mlir::emitc::CppEmitter &emitter) {
  llvm::raw_ostream &output = emitter.ostream();

  for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
    std::string buffer_name =
        moduleOp.getName().str() + "_" + rodataOp.getName().str();
    output << "iree_vm_buffer_initialize("
           << "IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE, "
           << "iree_make_byte_span("
           << "(void*)" << buffer_name << ", sizeof(" << buffer_name << ")), "
           << "iree_allocator_null(), "
           << "&state->rodata_buffers[" << rodataOp.ordinal() << "]"
           << ");\n";
  }

  return success();
}

static LogicalResult buildModuleDescriptors(IREE::VM::ModuleOp &moduleOp,
                                            mlir::emitc::CppEmitter &emitter) {
  SymbolTable symbolTable(moduleOp);
  std::string moduleName = moduleOp.getName().str();
  llvm::raw_ostream &output = emitter.ostream();

  auto printCStringView = [](StringRef s) -> std::string {
    return ("iree_make_cstring_view(\"" + s + "\")").str();
  };

  // exports
  std::string exportName = moduleName + "_exports_";
  output << "static const iree_vm_native_export_descriptor_t " << exportName
         << "[] = {\n";

  // sort export ops
  SmallVector<IREE::VM::ExportOp, 4> exportOps(
      moduleOp.getOps<IREE::VM::ExportOp>());
  llvm::sort(exportOps, [](auto &lhs, auto &rhs) {
    return lhs.export_name().compare(rhs.export_name()) < 0;
  });

  for (auto exportOp : exportOps) {
    StringRef funcName = exportOp.function_ref();
    auto funcOp = symbolTable.lookup<mlir::FuncOp>(funcName);
    if (!funcOp) {
      return exportOp.emitError("Couldn't find referenced FuncOp");
    }
    StringAttr callingConvention =
        funcOp.getOperation()->getAttr("calling_convention").cast<StringAttr>();
    if (!callingConvention) {
      return exportOp.emitError(
          "Couldn't create calling convention string for referenced FuncOp");
    }

    // TODO(simon-camp): support function-level reflection attributes
    output << "{" << printCStringView(exportOp.export_name()) << ", "
           << printCStringView(callingConvention.getValue()) << ", 0, NULL},\n";
  }
  output << "};\n";
  output << "\n";

  // imports
  std::string importName = moduleName + "_imports_";
  output << "static const iree_vm_native_import_descriptor_t " << importName
         << "[] = {\n";

  // sort import ops
  SmallVector<IREE::VM::ImportOp, 4> importOps(
      moduleOp.getOps<IREE::VM::ImportOp>());
  llvm::sort(importOps, [](auto &lhs, auto &rhs) {
    return lhs.getName().compare(rhs.getName()) < 0;
  });

  for (auto importOp : importOps) {
    output << "{" << printCStringView(importOp.getName()) << "},\n";
  }
  output << "};\n";
  output << "\n";

  // functions
  std::string functionName = moduleName + "_funcs_";
  output << "static const iree_vm_native_function_ptr_t " << functionName
         << "[] = {\n";

  // We only add exported functions to the table, as calls to internal functions
  // are directly mapped to C function calls of the generated implementation.
  for (auto exportOp : exportOps) {
    StringRef funcName = exportOp.function_ref();
    auto funcOp = symbolTable.lookup<mlir::FuncOp>(funcName);
    if (!funcOp) {
      return exportOp.emitError("Couldn't find referenced FuncOp");
    }
    output << "{"
           << "(iree_vm_native_function_shim_t)";

    if (failed(printShim(funcOp, output))) {
      return funcOp.emitError("Couldn't create calling convention string");
    }
    output << ", "
           << "(iree_vm_native_function_target_t)" << funcName << "},\n";
  }
  output << "};\n";
  output << "\n";

  // module descriptor
  // TODO(simon-camp): support module-level reflection attributes
  std::string descriptorName = moduleName + "_descriptor_";
  output << "static const iree_vm_native_module_descriptor_t " << descriptorName
         << " = {\n"
         << printCStringView(moduleName) << ",\n"
         << "IREE_ARRAYSIZE(" << importName << "),\n"
         << importName << ",\n"
         << "IREE_ARRAYSIZE(" << exportName << "),\n"
         << exportName << ",\n"
         << "IREE_ARRAYSIZE(" << functionName << "),\n"
         << functionName << ",\n"
         << "0,\n"
         << "NULL,\n"
         << "};\n";

  // destroy
  // TODO(simon-camp):

  // alloc_state
  output << "static iree_status_t " << moduleName
         << "_alloc_state(void* self, iree_allocator_t allocator, "
            "iree_vm_module_state_t** out_module_state) {\n"
         << moduleName << "_state_t* state = NULL;\n"
         << "IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, "
            "sizeof(*state), (void**)&state));\n "
         << "memset(state, 0, sizeof(*state));\n"
         << "state->allocator = allocator;\n";

  // initialize globals
  if (failed(initializeState(moduleOp, emitter))) {
    return moduleOp.emitError() << "Failed to emit state members";
  }

  output << "*out_module_state = (iree_vm_module_state_t*)state;\n"
         << "return iree_ok_status();\n"
         << "}\n";

  // free_state
  output << "static void " << moduleName
         << "_free_state(void* self, iree_vm_module_state_t* "
            "module_state) {\n"
         << moduleName << "_state_t* state = (" << moduleName
         << "_state_t*)module_state;\n"
         << "iree_allocator_free(state->allocator, state);\n"
         << "}\n";

  // resolve_imports
  output << "static iree_status_t " << moduleName << "_resolve_import("
         << "void* self, iree_vm_module_state_t* module_state, "
            "iree_host_size_t ordinal, const iree_vm_function_t* function, "
            "const iree_vm_function_signature_t* signature) {\n"
         << moduleName << "_state_t* state = (" << moduleName
         << "_state_t*)module_state;\n"
         << "state->imports[ordinal] = *function;\n"
         << "return iree_ok_status();\n}";

  output << "\n";

  // create
  output << "static iree_status_t " << moduleName << "_create("
         << "iree_allocator_t allocator, iree_vm_module_t** "
            "out_module) {\n"
         << "iree_vm_module_t interface;\n"
         << "IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, "
            "NULL));\n"
         << "interface.destroy = NULL;\n"
         << "interface.alloc_state = " << moduleName << "_alloc_state;\n"
         << "interface.free_state = " << moduleName << "_free_state;\n"
         << "interface.resolve_import = " << moduleName << "_resolve_import;\n"
         << "return iree_vm_native_module_create(&interface, "
            "&"
         << descriptorName << ", allocator, out_module);\n"
         << "}\n";

  output << "\n";
  return success();
}

/// Adapted from BytecodeModuleTarget and extended by C specific passes
static LogicalResult canonicalizeModule(
    IREE::VM::ModuleOp moduleOp, IREE::VM::CTargetOptions targetOptions) {
  OwningRewritePatternList patterns(moduleOp.getContext());
  ConversionTarget target(*moduleOp.getContext());
  target.addLegalDialect<IREE::VM::VMDialect>();
  target.addLegalOp<IREE::DoNotOptimizeOp>();

  // Add all VM canonicalization patterns and mark pseudo-ops illegal.
  auto *context = moduleOp.getContext();
  for (auto *op : context->getRegisteredOperations()) {
    // Non-serializable ops must be removed prior to serialization.
    if (op->hasTrait<OpTrait::IREE::VM::PseudoOp>()) {
      op->getCanonicalizationPatterns(patterns, context);
      target.setOpAction(OperationName(op->name, context),
                         ConversionTarget::LegalizationAction::Illegal);
    }

    // Debug ops must not be present when stripping.
    // TODO(benvanik): add RemoveDisabledDebugOp pattern.
    if (op->hasTrait<OpTrait::IREE::VM::DebugOnly>() &&
        targetOptions.stripDebugOps) {
      target.setOpAction(OperationName(op->name, context),
                         ConversionTarget::LegalizationAction::Illegal);
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

  modulePasses.addPass(createDropCompilerHintsPass());

  if (failed(passManager.run(moduleOp->getParentOfType<mlir::ModuleOp>()))) {
    return moduleOp.emitError() << "failed during transform passes";
  }

  return success();
}

LogicalResult translateModuleToC(IREE::VM::ModuleOp moduleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output) {
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
  printInclude("iree/vm/value.h");
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

    if (failed(emitter.emitTypes(*funcOp.getOperation(),
                                 funcOp.getType().getResults())))
      return failure();
    output << " " << funcOp.getName();

    output << "(";

    bool error = false;
    llvm::interleaveComma(
        funcOp.getArguments(), output, [&](BlockArgument arg) {
          if (failed(emitter.emitType(*funcOp.getOperation(), arg.getType())))
            error = true;
        });
    if (error) return failure();
    output << ");\n";
  }

  output << "// DEFINE FUNCTIONS\n";

  for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
    Operation *op = funcOp.getOperation();
    if (op->hasAttr("emitc.static")) output << "static ";
    if (failed(emitter.emitOperation(*funcOp.getOperation(),
                                     /*trailingSemicolon=*/
                                     false)))
      return failure();
  }

  printSeparatingComment(output);

  printModuleComment(moduleOp, output);
  output << "\n";

  // generate module descriptors
  if (failed(buildModuleDescriptors(moduleOp, emitter))) {
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
