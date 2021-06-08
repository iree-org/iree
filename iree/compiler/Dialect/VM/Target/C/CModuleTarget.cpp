// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"

#include "emitc/Target/Cpp/Cpp.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"
#include "iree/compiler/Dialect/VM/Target/CallingConventionUtils.h"
#include "iree/compiler/Dialect/VM/Target/ConstantEncodingUtils.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

static std::string buildFunctionName(IREE::VM::ModuleOp &moduleOp,
                                     IREE::VM::FuncOp &funcOp,
                                     bool implSuffix) {
  std::string functionName =
      std::string(moduleOp.getName()) + "_" + std::string(funcOp.getName());

  return implSuffix ? functionName + "_impl" : functionName;
}

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

  output << "struct " << moduleName << "_t;\n";
  output << "struct " << moduleName << "_state_t {\n";

  output << "iree_allocator_t allocator;\n";
  output << "uint8_t rwdata["
         << moduleOp.ordinal_counts().getValue().global_bytes() << "];\n";
  output << "iree_vm_ref_t refs["
         << moduleOp.ordinal_counts().getValue().global_refs() << "];\n";
  output << "iree_vm_buffer_t rodata_buffers["
         << moduleOp.ordinal_counts().getValue().rodatas() << "];\n";
  output << "};\n";

  output << "typedef struct " << moduleName << "_t " << moduleName << "_t;\n";
  output << "typedef struct " << moduleName << "_state_t " << moduleName
         << "_state_t;\n";

  output << "\n";

  return success();
}

static LogicalResult printShim(IREE::VM::FuncOp &funcOp,
                               llvm::raw_ostream &output) {
  auto callingConvention = makeCallingConventionString(funcOp);
  if (!callingConvention) {
    return funcOp.emitError("Couldn't create calling convention string");
  }
  output << "call_" << callingConvention.getValue() << "_shim";
  return success();
}

static LogicalResult printFuncOpArguments(IREE::VM::FuncOp &funcOp,
                                          mlir::emitc::CppEmitter &emitter) {
  return mlir::emitc::interleaveCommaWithError(
      funcOp.getArguments(), emitter.ostream(), [&](auto arg) -> LogicalResult {
        if (failed(emitter.emitType(*funcOp.getOperation(), arg.getType()))) {
          return failure();
        }
        emitter.ostream() << " " << emitter.getOrCreateName(arg);
        return success();
      });
}

// Function results get propagated through pointer arguments
static LogicalResult printFuncOpResults(
    IREE::VM::FuncOp &funcOp, mlir::emitc::CppEmitter &emitter,
    SmallVector<std::string, 4> &resultNames) {
  return mlir::emitc::interleaveCommaWithError(
      llvm::zip(funcOp.getType().getResults(), resultNames), emitter.ostream(),
      [&](std::tuple<Type, std::string> tuple) -> LogicalResult {
        Type type = std::get<0>(tuple);
        std::string resultName = std::get<1>(tuple);

        if (failed(emitter.emitType(*funcOp.getOperation(), type))) {
          return failure();
        }
        emitter.ostream() << " *" << resultName;
        return success();
      });
}

static LogicalResult initializeState(IREE::VM::ModuleOp moduleOp,
                                     mlir::emitc::CppEmitter &emitter) {
  llvm::raw_ostream &output = emitter.ostream();

  for (auto globalOp : moduleOp.getOps<IREE::VM::GlobalI32Op>()) {
    Optional<Attribute> initialValue = globalOp.initial_value();
    Optional<StringRef> initializer = globalOp.initializer();
    if (initialValue.hasValue()) {
      // TODO(simon-camp): We can't represent structs in emitc (yet maybe), so
      // the struct argument name here must not be changed.
      emitter.ostream() << "vm_global_store_i32(state->rwdata, "
                        << globalOp.ordinal() << ", ";
      if (failed(emitter.emitAttribute(*globalOp.getOperation(),
                                       initialValue.getValue()))) {
        return globalOp.emitError() << "Unable to emit initial_value";
      }
      emitter.ostream() << ");\n";
    } else if (initializer.hasValue()) {
      return globalOp.emitError()
             << "Initializers for globals not supported yet";
    }
  }
  // TODO(simon-camp): Support globals with different element type

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

static LogicalResult translateBranchOp(IREE::VM::BranchOp branchOp,
                                       mlir::emitc::CppEmitter &emitter) {
  auto &output = emitter.ostream();
  Block &successor = *branchOp.getSuccessor();

  for (auto pair :
       llvm::zip(branchOp.getOperands(), successor.getArguments())) {
    auto &operand = std::get<0>(pair);
    auto &argument = std::get<1>(pair);
    output << emitter.getOrCreateName(argument) << " = "
           << emitter.getOrCreateName(operand) << ";\n";
  }

  output << "goto ";
  if (!(emitter.hasBlockLabel(successor))) {
    return branchOp.emitOpError() << "Unable to find label for successor block";
  }
  output << emitter.getOrCreateName(successor) << ";\n";
  return success();
}

static LogicalResult translateCallOpToC(IREE::VM::CallOp callOp,
                                        mlir::emitc::CppEmitter &emitter) {
  return success();
}

static LogicalResult translateCondBranchOp(IREE::VM::CondBranchOp condBranchOp,
                                           mlir::emitc::CppEmitter &emitter) {
  llvm::raw_ostream &output = emitter.ostream();

  Block &trueSuccessor = *condBranchOp.getTrueDest();
  Block &falseSuccessor = *condBranchOp.getFalseDest();

  output << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
         << ") {\n";

  // If condition is true.
  for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
                             trueSuccessor.getArguments())) {
    auto &operand = std::get<0>(pair);
    auto &argument = std::get<1>(pair);
    output << emitter.getOrCreateName(argument) << " = "
           << emitter.getOrCreateName(operand) << ";\n";
  }

  output << "goto ";
  if (!(emitter.hasBlockLabel(trueSuccessor))) {
    return condBranchOp.emitOpError()
           << "Unable to find label for successor block";
  }
  output << emitter.getOrCreateName(trueSuccessor) << ";\n";
  output << "} else {\n";
  // If condition is false.
  for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
                             falseSuccessor.getArguments())) {
    auto &operand = std::get<0>(pair);
    auto &argument = std::get<1>(pair);
    output << emitter.getOrCreateName(argument) << " = "
           << emitter.getOrCreateName(operand) << ";\n";
  }

  output << "goto ";
  if (!(emitter.hasBlockLabel(falseSuccessor))) {
    return condBranchOp.emitOpError()
           << "Unable to find label for successor block";
  }
  output << emitter.getOrCreateName(falseSuccessor) << ";\n";
  output << "}\n";
  return success();
}

static LogicalResult translateFailOp(IREE::VM::FailOp failOp,
                                     mlir::emitc::CppEmitter &emitter,
                                     bool hasRefs) {
  llvm::raw_ostream &output = emitter.ostream();

  auto status = failOp.status();

  if (hasRefs) {
    output << "VM_REF_ARRAY_RELEASE(local_refs);\n";
  }

  output << "return vm_fail_or_ok(" << emitter.getOrCreateName(status)
         << ", iree_make_cstring_view(\"" << failOp.message() << "\"));\n";
  return success();
}

static LogicalResult translateReturnOpToC(
    IREE::VM::ReturnOp returnOp, mlir::emitc::CppEmitter &emitter,
    SmallVector<std::string, 4> resultNames, bool hasRefs) {
  llvm::raw_ostream &output = emitter.ostream();

  for (std::tuple<Value, std::string> tuple :
       llvm::zip(returnOp.getOperands(), resultNames)) {
    Value operand = std::get<0>(tuple);
    std::string resultName = std::get<1>(tuple);
    output << "*" << resultName << " = " << emitter.getOrCreateName(operand)
           << ";\n";
  }

  if (hasRefs) {
    output << "VM_REF_ARRAY_RELEASE(local_refs);\n";
  }

  output << "return iree_ok_status();\n";

  return success();
}

static LogicalResult translateOpToC(Operation &op,
                                    mlir::emitc::CppEmitter &emitter,
                                    SmallVector<std::string, 4> resultNames,
                                    bool hasRefs) {
  if (auto branchOp = dyn_cast<IREE::VM::BranchOp>(op))
    return translateBranchOp(branchOp, emitter);
  if (auto callOp = dyn_cast<IREE::VM::CallOp>(op))
    return translateCallOpToC(callOp, emitter);
  if (auto condBranchOp = dyn_cast<IREE::VM::CondBranchOp>(op))
    return translateCondBranchOp(condBranchOp, emitter);
  if (auto failOp = dyn_cast<IREE::VM::FailOp>(op))
    return translateFailOp(failOp, emitter, hasRefs);
  if (auto returnOp = dyn_cast<IREE::VM::ReturnOp>(op))
    return translateReturnOpToC(returnOp, emitter, resultNames, hasRefs);
  // Fall back to generic emitc printer
  if (succeeded(emitter.emitOperation(op, /*trailingSemicolon=*/true))) {
    return success();
  }

  return failure();
}

static LogicalResult translateFunctionToC(IREE::VM::ModuleOp &moduleOp,
                                          IREE::VM::FuncOp &funcOp,
                                          mlir::emitc::CppEmitter &emitter) {
  std::string moduleName = moduleOp.getName().str();
  emitc::CppEmitter::Scope scope(emitter);
  llvm::raw_ostream &output = emitter.ostream();

  // this function later gets wrapped with argument marshalling code
  std::string functionName =
      buildFunctionName(moduleOp, funcOp, /*implSuffix=*/true);

  output << "iree_status_t " << functionName << "(";

  if (failed(printFuncOpArguments(funcOp, emitter))) {
    return failure();
  }

  if (funcOp.getNumResults() > 0 && funcOp.getNumArguments() > 0) {
    output << ", ";
  }

  SmallVector<std::string, 4> resultNames;
  for (unsigned int idx = 0; idx < funcOp.getNumResults(); idx++) {
    std::string resultName = "out" + std::to_string(idx);
    resultNames.push_back(resultName);
  }

  if (failed(printFuncOpResults(funcOp, emitter, resultNames))) {
    return failure();
  }

  if (funcOp.getNumArguments() + funcOp.getNumResults() > 0) {
    output << ", ";
  }

  // TODO(simon-camp): We can't represent structs in emitc (yet maybe), so the
  // struct argument name here must not be changed.
  output << moduleName << "_state_t* state) {\n";

  // We forward declare all result variables except for the ones with RefType.
  output << "// VARIABLE DECLARATIONS\n";
  output << "// RESULTS\n";
  for (auto &op : funcOp.getOps()) {
    for (auto result : op.getResults()) {
      if (result.getType().isa<IREE::VM::RefType>()) {
        continue;
      }
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true))) {
        return op.emitError() << "Unable to declare result variable for op";
      }
    }
  }
  output << "// BASIC BLOCK ARGUMENTS\n";

  auto &blocks = funcOp.getBlocks();
  // Create label names for basic blocks.
  for (auto &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Emit variables for basic block arguments (omitting the first).
  for (auto it = std::next(blocks.begin()); it != blocks.end(); ++it) {
    Block &block = *it;
    for (auto &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg)) {
        // This shouldn't happen
        return failure();
      }
      if (failed(emitter.emitType(*funcOp.getOperation(), arg.getType()))) {
        return failure();
      }
      output << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  output << "// END VARIABLE DECLARATIONS\n";

  // We reuse the register allocation pass and emit an array for all Values with
  // ref type instead of generating one variable per Value. This makes the
  // deallocation process easier for us.
  RegisterAllocation registerAllocation;
  if (failed(registerAllocation.recalculate(funcOp))) {
    return funcOp.emitOpError() << "unable to perform register allocation";
  }

  const size_t numRefs = registerAllocation.getMaxRefRegisterOrdinal() + 1;
  const bool hasRefs = numRefs > 0;

  if (hasRefs) {
    auto ref_initializers = SmallVector<StringRef, 4>{numRefs, "{0}"};
    output << "iree_vm_ref_t local_refs[" << numRefs << "] = {"
           << llvm::join(ref_initializers, ", ") << "};\n";
  }

  for (auto &block : blocks) {
    // Only print a label if there is more than one block.
    if (blocks.size() > 1) {
      if (failed(emitter.emitLabel(block))) {
        return funcOp.emitOpError() << "Unable to print label for basic block";
      }
    }
    for (Operation &op : block.getOperations()) {
      if (failed(
              translateOpToC(op, emitter, resultNames, /*hasRefs=*/hasRefs))) {
        return failure();
      }
    }
  }

  output << "}\n";

  return success();
}

static LogicalResult buildModuleDescriptors(IREE::VM::ModuleOp &moduleOp,
                                            mlir::emitc::CppEmitter &emitter) {
  SymbolTable symbolTable(moduleOp);
  std::string moduleName = moduleOp.getName().str();
  llvm::raw_ostream &output = emitter.ostream();

  // function wrapper
  for (auto funcOp : moduleOp.getOps<IREE::VM::FuncOp>()) {
    output << "static iree_status_t "
           << buildFunctionName(moduleOp, funcOp,
                                /*implSufffix=*/false)
           << "("
           << "iree_vm_stack_t* stack, " << moduleName << "_t* module, "
           << moduleName << "_state_t* state";

    if (funcOp.getNumArguments() > 0) {
      output << ", ";
    }

    if (failed(printFuncOpArguments(funcOp, emitter))) {
      return failure();
    }

    if (funcOp.getNumArguments() > 0) {
      output << ", ";
    }

    SmallVector<std::string, 4> resultNames;
    for (unsigned int idx = 0; idx < funcOp.getNumResults(); idx++) {
      std::string resultName = "out" + std::to_string(idx);
      resultNames.push_back(resultName);
    }

    if (failed(printFuncOpResults(funcOp, emitter, resultNames))) {
      return failure();
    }
    output << ") {\n"
           << "return "
           << buildFunctionName(moduleOp, funcOp,
                                /*implSufffix=*/true)
           << "(";

    SmallVector<std::string, 4> argNames;
    for (Value &argument : funcOp.getArguments()) {
      std::string argName = emitter.getOrCreateName(argument).str();
      argNames.push_back(argName);
    }

    output << llvm::join(argNames, ", ");

    if (funcOp.getNumResults() > 0) {
      output << ", ";
    }

    output << llvm::join(resultNames, ", ");

    if (funcOp.getNumArguments() + funcOp.getNumResults() > 0) {
      output << ", ";
    }
    output << "state);\n}\n";
  }

  auto printCStringView = [](std::string s) -> std::string {
    return "iree_make_cstring_view(\"" + s + "\")";
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
    auto funcOp = symbolTable.lookup<IREE::VM::FuncOp>(exportOp.function_ref());
    if (!funcOp) {
      return exportOp.emitError("Couldn't find referenced FuncOp");
    }
    auto callingConvention = makeCallingConventionString(funcOp);
    if (!callingConvention) {
      return exportOp.emitError(
          "Couldn't create calling convention string for referenced FuncOp");
    }

    // TODO(simon-camp): support function-level reflection attributes
    output << "{" << printCStringView(exportOp.export_name().str()) << ", "
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
    output << "{" << printCStringView(importOp.getName().str()) << "},\n";
  }
  output << "};\n";
  output << "\n";

  // functions
  std::string functionName = moduleName + "_funcs_";
  output << "static const iree_vm_native_function_ptr_t " << functionName
         << "[] = {\n";

  // sort func ops
  SmallVector<IREE::VM::FuncOp, 4> funcOps(moduleOp.getOps<IREE::VM::FuncOp>());
  llvm::sort(funcOps, [&moduleOp](auto &lhs, auto &rhs) {
    std::string lhsStr =
        buildFunctionName(moduleOp, lhs, /*implSufffix=*/false);
    std::string rhsStr =
        buildFunctionName(moduleOp, rhs, /*implSufffix=*/false);
    return lhsStr.compare(rhsStr) < 0;
  });
  for (auto funcOp : funcOps) {
    output << "{"
           << "(iree_vm_native_function_shim_t)";

    if (failed(printShim(funcOp, output))) {
      return funcOp.emitError("Couldn't create calling convention string");
    }
    output << ", "
           << "(iree_vm_native_function_target_t)"
           << buildFunctionName(moduleOp, funcOp, /*implSufffix=*/false)
           << "},\n";
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
  // TODO(simon-camp):

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
         << "interface.resolve_import = NULL;\n"
         << "return iree_vm_native_module_create(&interface, "
            "&"
         << descriptorName << ", allocator, out_module);\n"
         << "}\n";

  output << "\n";
  return success();
}

// Adapted from BytecodeModuleTarget and extended by C specific passes
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

  // In the the Bytecode module the order is:
  // * `createDropCompilerHintsPass()`
  // * `IREE::VM::createOrdinalAllocationPass()`
  // Here, we have to reverse the order and run `createConvertVMToEmitCPass()`
  // inbetween to test the EmitC pass. Otherwise, the constants get folded
  // by the canonicalizer.

  // Mark up the module with ordinals for each top-level op (func, etc).
  // This will make it easier to correlate the MLIR textual output to the
  // binary output.
  // We don't want any more modifications after this point as they could
  // invalidate the ordinals.
  modulePasses.addPass(IREE::VM::createOrdinalAllocationPass());

  // C target specific passes
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
  printInclude("iree/vm/shims_emitc.h");
  printInclude("iree/vm/value.h");
  output << "\n";

  printModuleComment(moduleOp, output);
  output << "\n";

  mlir::emitc::CppEmitter emitter(output, /*restrictToC=*/true,
                                  /*forwardDeclareVariables=*/true);
  mlir::emitc::CppEmitter::Scope scope(emitter);

  if (failed(printRodataBuffers(moduleOp, emitter))) {
    return failure();
  }

  // build struct definitions
  if (failed(printStructDefinitions(moduleOp, emitter))) {
    return failure();
  }

  // translate functions
  for (auto funcOp : moduleOp.getOps<IREE::VM::FuncOp>()) {
    if (failed(translateFunctionToC(moduleOp, funcOp, emitter))) {
      return failure();
    }

    output << "\n";
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
